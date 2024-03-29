import math
import re
import numpy as np
import os
import json 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


###########################################---------PROCESAREA DATELOR---------###########################################

# functie de prelucrate a datelor de intrare intr-un format mai eficient pentru lucru
# pentru un fisier de date de intrare se determina atributele (cuvintele din email-uri) (= data["features"])
# pentru datele de antrenament (part 1- 9) pnetru fiecare email se construieste un "vector de adiacenta" = un vector ce contine indicii atributelor care apar in email (= data["X"])
# pentru toate datele de antrenament se construieste un vector de clase (= data["Y"])
# la final aceste date procesate sunt salvate in data.txt, intr-un fiseir separata pentru fiecare grup de date (bare, lemm, lemm_stop, stop) 
def get_features(directory):
    dir_processed = {}
    data_directory = os.path.join(".\\database", directory)
    try:
        os.mkdir(data_directory)
    except:
        pass
    processed_file = open(os.path.join(data_directory, "data.txt"), 'w')

    main_dir = ".\\lingspam_public"
    root_dir = os.path.join(main_dir, directory)
    parts = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    parts = sorted(parts, key=lambda x: int(x.split('part')[-1]))

    words = set()
    for part in parts:
        emails = [os.path.join(part, e) for e in os.listdir(part)]
        for e in emails:
            email = open(e, 'r')
            lines = email.readlines()
            content = lines[0] + lines[2]
            for w in content.split():
                if w.isalpha() and len(w) > 1:
                    words.add(w.lower())

    dir_processed["features"] = list(words)

    dir_processed["X"] = []
    dir_processed["Y"] = []
    count = 0
    
    for part in parts[:-1]:
        emails = [os.path.join(part, e) for e in os.listdir(part)]
        for e in emails:
            count += 1
            feat = []
            email = open(e, 'r').readlines()
            content = email[0].split() + email[2].split()
            words = [w for w in content if w.isalpha() and len(w) > 1]
            words = list(set(words))
            for w in words:
                index = dir_processed["features"].index(w)
                feat.append(index)
            dir_processed["X"].append(feat)
            dir_processed["Y"].append(0 if re.search('spmsg*', e) else 1)

    dir_processed["count"] = count

    processed_file.write(json.dumps(dir_processed))

    
# functie care determina din datele prelucrate mai sus o tablea de probabilitati pentru calculele din algoritmul bayes
# se calculeaza P(spam) si P(safe)
# pentru fiecare cuvant w se calculeaza P(w|safe) si P(w|spam)
# toate probabilitatile se salveza intr-un dictionary care este apoi salvat in fisierul probability.txt, unul pentru fiecare fiseir de antrenament in parte
def calculate_probability_table(directory):
    alpha = 1
    dir_probability = {}
    data_directory = os.path.join(".\\database", directory)

    probability_file = open(os.path.join(data_directory, "probability.txt"), 'w')

    data_file = open(os.path.join(data_directory, "data.txt"), 'r')
    data = json.loads(data_file.read())

    safe_index = [i for i in range(0, data["count"]) if data["Y"][i] == 1]
    spam_index = [i for i in range(0, data["count"]) if data["Y"][i] == 0]

    count_safe = len(safe_index)
    count_spam = len(spam_index)

    safe_probability = count_safe/data["count"]
    spam_probability = count_spam/data["count"]

    dir_probability["P(safe)"] = safe_probability
    dir_probability["P(spam)"] = spam_probability

    l = len(data["features"])

    probab_safe = np.zeros(l) + alpha
    probab_spam = np.zeros(l) + alpha

    i = 0 
    dir_probability["P(w|safe)"] = []
    dir_probability["P(w|spam)"] = []
    for email in data["X"]:
        if i in safe_index:
            for index in email:
                probab_safe[index] += 1
        else:
            for index in email:
                probab_spam[index] += 1
        i += 1

    for i in range(l):
        dir_probability["P(w|safe)"].append(probab_safe[i]/(count_safe + alpha * l))
        dir_probability["P(w|spam)"].append(probab_spam[i]/(count_safe + alpha * l))

    probability_file.write(json.dumps(dir_probability))


# ruleaza functiile pt fiecare directoriu
# dupa rulare se obtine fisierul database care contine toate datele procesate pentru acces mai eficient
def process_all_data():
    root_dir = ".\\lingspam_public"
    dirs = os.listdir(root_dir)
    for dir in dirs:
        if dir != 'readme.txt':
            get_features(dir)
            calculate_probability_table(dir)

# process_all_data()    # ERT = 600 sec


###########################################---------ALGORITMUL BAYES NAIV---------###########################################


# fuctia implementeaza algoritmul bayes naiv pentru clasificarea unei instante de test
# foloseste formula principala a algoritmului pentru a calcula p_safe si p_spam, clasificarea finala fiind cea mai mare ca valoare
# pentru ca algoritmul lucreaza cu produsul mai multor valor mici am folosit log(probability) pentru a nu obtine 0 pentru ambele p-uri
def bayes_naiv_clasifier(dir, email_file):
    email = open(email_file, 'r').readlines()
    e = email[0].split() + email[2].split()
    content = [w for w in e if (w.isalpha() and len(w) > 1)]
    
    data_directory = os.path.join(".\\database", dir)

    probability_file = open(os.path.join(data_directory, "probability.txt"), 'r')
    probabilitys = json.loads(probability_file.read())
    
    data_file = open(os.path.join(data_directory, "data.txt"), 'r')
    data = json.loads(data_file.read())

    words = data["features"]

    prob_safe = probabilitys["P(w|safe)"]
    prob_spam = probabilitys["P(w|spam)"]

    p_safe = probabilitys["P(safe)"]
    p_spam = probabilitys["P(spam)"]

    for i in range(len(words)):
        if content.count(words[i]) != 0 and prob_safe[i] != 0 and prob_spam[i] != 0:
            p_safe += math.log(prob_safe[i])
            p_spam += math.log(prob_spam[i])
        elif content.count(words[i]) == 0 and prob_safe[i] != 0 and prob_spam[i] != 0:
            p_safe += math.log(1 - prob_safe[i])
            p_spam += math.log(1 - prob_spam[i])

    return p_safe, p_spam

# functia calculeaza acuratetea de test, folosind part10 ca date de test
# ERT = 500 sec / directory
def test_accuracy(dir):
    root_dir = ".\\lingspam_public"
    true_value = []
    test_y = []
    part = root_dir + "\\" + dir + "\\part10"
    for email in os.listdir(part):
        result = bayes_naiv_clasifier(dir, part + "\\" + email)
        test_y.append(np.argmax(result))
        true_value.append(0 if not re.search('spmsg*', email) else 1)

    return accuracy_score(true_value, test_y)

# print(f"Accuracy score :  {test_accuracy('bare')}")


###########################################---------CVLOO---------###########################################

# functie auxiliara pentru implementarea cvloo
# folsita pentru recalcularea probabilitatilor folosite la bayes naiv prin redefinirea probabilitatilor P(w|safe) si P(w|spam) pentru cuvintele din emailul scos din antrenament
# functia returneaza probabilitatile recalculate
def redefine_probabilities(dir, email, type):
    data_directory = os.path.join(".\\database", dir)

    probability_file = open(os.path.join(data_directory, "probability.txt"), 'r')
    probabilitys = json.loads(probability_file.read())
    
    data_file = open(os.path.join(data_directory, "data.txt"), 'r')
    data = json.loads(data_file.read())

    if type == 1:
        safe_count = len([i for i in range(0, data["count"]) if data["Y"][i] == 1])
        for index in email:
            probab = probabilitys["P(w|safe)"][index]
            probab *= safe_count
            probab -= 1
            probab /= safe_count
            probabilitys["P(w|safe)"][index] = probab
    else:
        spam_count = len([i for i in range(0, data["count"]) if data["Y"][i] == 0])
        for index in email:
            probab = probabilitys["P(w|spam)"][index]
            probab *= spam_count
            probab -= 1
            probab /= spam_count
            probabilitys["P(w|spam)"][index] = probab

    return probabilitys


# functia de clasificarea a unei instante scoase prin cvloo
# functia incepe prin rularea functiei de calcul a probabilitatilor cu instanta scoasa si continua cu calculul p_safe si p_spam, clasificarea finala fiind cea mai mare ca valoare
def bayes_naiv_clasifier_cvloo(dir, instance, type):
    probabilitys = redefine_probabilities(dir, instance, type)
    
    data_directory = os.path.join(".\\database", dir)
    data_file = open(os.path.join(data_directory, "data.txt"), 'r')
    data = json.loads(data_file.read())

    words = data["features"]

    prob_safe = probabilitys["P(w|safe)"]
    prob_spam = probabilitys["P(w|spam)"]

    p_safe = probabilitys["P(safe)"]
    p_spam = probabilitys["P(spam)"]

    for i in range(len(words)):
        epsilon = 1e-10
        prob_safe_i = max(prob_safe[i], epsilon)
        prob_spam_i = max(prob_spam[i], epsilon)

        if i in instance:
            p_safe += math.log(prob_safe_i)
            p_spam += math.log(prob_spam_i)
        elif i not in instance:
            p_safe += math.log(1 - prob_safe_i)
            p_spam += math.log(1 - prob_spam_i)

    return p_safe, p_spam

# functie de calcul a acuratetei cvloo
# pentru fiecare instanta din cele de antrenament se apeleaza bayes_naiv_clasifier_cvloo si returneaza acuratetea totala pe setul de date 
# ERT = 900 sec / directory
def cross_validate_cvloo(dir):
    data_directory = os.path.join(".\\database", dir)
    data_file = open(os.path.join(data_directory, "data.txt"), 'r')
    data = json.loads(data_file.read())

    accuracies = []

    for i in range(data["count"]):
        test_instance = {"X": data["X"][i], "Y": data["Y"][i]}

        result = bayes_naiv_clasifier_cvloo(dir, test_instance['X'], test_instance["Y"])

        accuracy = 1 if np.argmax(result) == test_instance["Y"] else 0
        accuracies.append(accuracy)

    overall_accuracy = np.mean(accuracies)
    return overall_accuracy


# print(f"Overall accuracy with cvloo: {cross_validate_cvloo("bare")}")


###########################################---------GARFIC DE PERFORMANTA---------###########################################


def plot_accuracies():
    accuracies = {}
    directories = ['bare', 'lemm', 'lemm_stop', 'stop']
    for dir in directories:
        accuracies[dir] = {}
        accuracies[dir]['random'] = 0.7
        accuracies[dir]['test accuracy'] = test_accuracy(dir)
        accuracies[dir]['cvloo accuracy'] = cross_validate_cvloo(dir)
        print(f"Accuracies for {dir} : {accuracies[dir]}")

    bar_width = 0.25
    index = range(4)
    plt.bar(index, [val['random'] for val in accuracies.values()], width=bar_width, label='Random')
    plt.bar([i + bar_width for i in index], [val['test accuracy'] for val in accuracies.values()], width=bar_width, label='Test Accuracy')
    plt.bar([i + 2 * bar_width for i in index], [val['cvloo accuracy'] for val in accuracies.values()], width=bar_width, label='CVLOO Accuracy')

    plt.xlabel('Scenarios')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Accuracies')
    plt.xticks([i + bar_width for i in index], directories)
    plt.legend()

    plt.show()

plot_accuracies()
