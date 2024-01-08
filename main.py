import re
import numpy as np
import os
import json 

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
    print(parts)

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

    dir_processed["featured"] = list(words)

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
                index = dir_processed["featured"].index(w)
                word_count = content.count(w)
                tup = (index, word_count)
                feat.append(tup)
            dir_processed["X"].append(feat)
            dir_processed["Y"].append(0 if re.search('spmsg*', e) else 1)
        print(part)

    dir_processed["count"] = count

    processed_file.write(json.dumps(dir_processed))

    
def calculate_probability_table(directory):
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

    l = len(data["featured"])

    probab_safe = np.zeros(l)
    probab_spam = np.zeros(l)

    i = 0 
    dir_probability["P(w|safe)"] = []
    dir_probability["P(w|spam)"] = []
    for email in data["X"]:
        if i in safe_index:
            for tup in email:
                probab_safe[tup[0]] += 1
        else:
            for tup in email:
                probab_spam[tup[0]] += 1
        i += 1

    for i in range(l):
        dir_probability["P(w|safe)"].append(probab_safe[i]/count_safe)
        dir_probability["P(w|spam)"].append(probab_spam[i]/count_spam)

    probability_file.write(json.dumps(dir_probability))


#ruleaza functiile pt fiecare directoriu
def process_all_data():
    root_dir = ".\\lingspam_public"
    dirs = os.listdir(root_dir)
    for dir in dirs:
        if dir != 'readme.txt':
            get_features(dir)
            calculate_probability_table(dir)

#process_all_data()



def bayes_naiv_clasifier(dir, email_file):
    email = open(email_file, 'r').readlines()
    e = email[0].split() + email[2].split()
    content = [w for w in e if (w.isalpha() and len(w) > 1)]
    
    data_directory = os.path.join(".\\database", dir)

    probability_file = open(os.path.join(data_directory, "probability.txt"), 'r')
    probabilitys = json.loads(probability_file.read())
    
    data_file = open(os.path.join(data_directory, "data.txt"), 'r')
    data = json.loads(data_file.read())

    words = data["featured"]

    prob_safe = probabilitys["P(w|safe)"]
    prob_spam = probabilitys["P(w|spam)"]
    p_safe = probabilitys["P(safe)"]
    p_spam = probabilitys["P(spam)"]

    for i in range(1, len(prob_safe)-1):
        if content.count(words[i-1]) != 0:
            p_safe *= prob_safe[i]
            p_spam *= prob_spam[i]
        else:
            p_safe *= 1-prob_safe[i]
            p_spam *= 1-prob_spam[i]

    return p_safe, p_spam


print(bayes_naiv_clasifier("bare", ".\\lingspam_public\\bare\\part10\\9-5msg1.txt"))