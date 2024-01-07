import re
import numpy as np
import os

#functie care construieste o lista de civinte unice care apar in email-urile din baza de date
#functia parcurge fiecare email din baza de date si adauga cuvintele intr-un set de elemente unice
def get_features():
    root_dir = ".\\lingspam_public"
    files = os.listdir(root_dir)
    parts = []
    for f in files:
        f_root = os.path.join(root_dir, f)
        if f != 'readme.txt':
            parts += [os.path.join(f_root, a) for a in os.listdir(f_root)]

    words = set()
    for dir in parts:
        emails = [os.path.join(dir, e) for e in os.listdir(dir)]
        for e in emails:
            email = open(e, 'r')
            lines = email.readlines()
            content = lines[0] + lines[2]
            for w in content.split():
                if w.isalpha() and len(w) > 1:
                    words.add(w.lower())

    return list(words)

#deoarece rularea functiei de mai sus care proceseaza toate cuvintele din email-uri dureaza 46.848 secunde vom scrie lista finala intr-un fisier pentu a fi accesat fara procesartea tuturol fisierleor la fiecare rulare
def write_list_in_file():
    file = open("words.txt", 'w')
    words = ' '.join(get_features())
    file.write(words)

#write_list_in_file()
    

#avem 10408 emailuri pentru train si 59836 feature -uri
#functia ia ficare cuvant ca atribut si ficare email instanta si contruieste tabelelul de date in train_x
#nu vom folosi aboardarea asta deoarece pentru cele 10000 de emailuri timpul de rulare va fi 1000 sec ~= 16 min si dimensiune fisierului final ~= 24000 MB
#am pus un punct de orpire la 300 email-uri pentru a dedea cum ar arata baza de date processsata in felul asta
def process_data_into_database1():
    root_dir = ".\\lingspam_public"
    files = os.listdir(root_dir)
    parts = []
    for f in files:
        f_root = os.path.join(root_dir, f)
        if f != 'readme.txt':
            parts += [os.path.join(f_root, a) for a in os.listdir(f_root) if a != 'part10']
    train_y = []
    at = 0 
    
    output = open("train_x old.csv", 'w')
    file = open("words.txt", 'r')
    collums = file.readline().split()

    for dir in parts:
        emails = [os.path.join(dir, e) for e in os.listdir(dir)]
        for e in emails:
            if at == 300:
                break
            at += 1
            features = np.zeros(len(collums))
            email = open(e, 'r').readlines()
            content = email[0].split() + email[2].split()
            for w in content:
                if w.isalpha() and len(w) > 1:
                    index = collums.index(w.lower())
                    if index != -1:
                        features[index] = content.count(w)
            output.write(' '.join(map(str, map(int, features))) + '\n')
            train_y.append(1 if re.search('spmsg*', e) else 0)

    return train_y


#train_y = process_data_into_database1()

#a doua abordare va implica salvarea unor tuple cu indexul si count-ul fiecarui cuvant dintr-un email. In tuplu indexul este pozitia cuvantului in vectorul de feature-uri si count-ul este individualizat pt fiecare email
#salvarea se va face tot intr-un csv pentru accesarea mai rapida dupa prima rulare 
def process_data_into_database2():
    root_dir = ".\\lingspam_public"
    files = os.listdir(root_dir)
    parts = []
    for f in files:
        f_root = os.path.join(root_dir, f)
        if f != 'readme.txt':
            parts += [os.path.join(f_root, a) for a in os.listdir(f_root) if a != 'part10']
    
    output = open("train_x.csv", 'w')
    file = open("words.txt", 'r')
    features = file.readline().split()

    for dir in parts:
        emails = [os.path.join(dir, e) for e in os.listdir(dir)]
        for e in emails:
            feat = []
            email = open(e, 'r').readlines()
            content = email[0].split() + email[2].split()
            words = [w for w in content if w.isalpha() and len(w) > 1]
            words = list(set(words))
            for w in words:
                index = features.index(w)
                count = content.count(w)
                tup = (index, count)
                feat.append(tup)
            output.write(' '.join(map(str, feat)) + '\n')

#process_data_into_database2()

def process_data_class():
    root_dir = ".\\lingspam_public"
    files = os.listdir(root_dir)
    parts = []
    for f in files:
        f_root = os.path.join(root_dir, f)
        if f != 'readme.txt':
            parts += [os.path.join(f_root, a) for a in os.listdir(f_root) if a != 'part10']

    train_y = []

    for dir in parts:
        emails = [os.path.join(dir, e) for e in os.listdir(dir)]
        for e in emails:
            train_y.append(1 if re.search('spmsg*', e) else 0)

    return train_y

train_y = process_data_class()
#pentru implementarea algoritmului bayes naiv vom avea nevoie de computarea probabilitatilor de tipul P(email are cuvantul W|email e/nu e spam)
#din aceasta computare se va forma un tabel de probabilitati avand cuvintele ca coloane si e/nu e spam ca linii
#functia de mai joi proceseaza aceste probabilitati pe baza tabeleului de date determinat mai sus 
#vom calcula doar probabilitatile de tipul p = P(email are cuvantul W|email e/nu e spam), nu si cazul in care emailul nu are cuvantul, acesta fiind 1-p
def process_conditional_probability():
    probability_table = open("probability_table.txt", 'w')

    file = open("words.txt", 'r')
    features = file.readline().split()

    emails = open("train_x.csv", 'r')

    safe_index = [i for i in range(0, len(train_y)) if train_y[i] == 1]
    spam_index = [i for i in range(0, len(train_y)) if train_y[i] == 0]

    count_safe = len(safe_index)
    count_spam = len(spam_index)

    safe_probability = count_safe/len(train_y)
    spam_probability = count_spam/len(train_y)

    safe_line = [safe_probability]
    spam_line = [spam_probability]

    probab_safe = np.zeros(len(features))
    probab_spam = np.zeros(len(features))


    i = 0 
    for email in emails.readlines():
        tuples_list = re.findall(r'\((.*?)\)', email)
        tuples_list = [tuple(map(int, tup.strip().split(','))) for tup in tuples_list]
        if i in safe_index:
            for tup in tuples_list:
                probab_safe[tup[0]] += 1
        else:
            for tup in tuples_list:
                probab_spam[tup[0]] += 1
        i += 1

    for i in range(len(features)):
        safe_line.append(probab_safe[i]/count_safe)
        spam_line.append(probab_spam[i]/count_spam)

    probability_table.write(' '.join(map(str, safe_line)) + '\n')
    probability_table.write(' '.join(map(str, spam_line)))


#process_conditional_probability()
#in final in probability table avem: 2 linii unde primul elemet din linia 1 e p(email = safe) si linia 2 p(email = safe) urmate fiecare respectiv de 59836 probabilitiati pentru fiecare cuvant gasit in email-uri
    

def bayes_naiv_clasifier(email_file):
    email = open(email_file, 'r').readlines()
    e = email[0].split() + email[2].split()
    content = [w for w in e if (w.isalpha() and len(w) > 1)]

    words = open("words.txt", 'r').readline().split()

    probabilitys = open("probability_table.txt", 'r').readlines()
    prob_safe = probabilitys[0].split()
    prob_spam = probabilitys[1].split()

    p_safe = float(prob_safe[0])
    p_spam = float(prob_spam[0])

    for i in range(1, len(prob_safe)-1):
        if content.count(words[i-1]) != 0:
            p_safe *= float(prob_safe[i])
            p_spam *= float(prob_spam[i])
        else:
            p_safe *= 1-float(prob_safe[i])
            p_spam *= 1-float(prob_spam[i])

    return p_safe, p_spam


safe, spam = bayes_naiv_clasifier(".\\lingspam_public\\lemm_stop\\part10\\9-5msg2.txt")

print((safe, spam))
