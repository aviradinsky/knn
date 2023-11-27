#!.e/bin/python3.12
# -*- coding: utf-8 -*-

import math
import pandas as pd
import numpy as np

e_ans = []
h_ans = []
m_ans = []


class DistClass:
    def __init__(self, dist=-1, tag="-"):
        self.dist = dist
        self.tag = tag


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        num1 = float(instance1[x])
        num2 = float(instance2[x])
        distance += pow(num1 - num2, 2)
    return math.sqrt(distance)


def hamming_distance(instance1, instance2, length):
    distance = sum(c1 != c2 for c1, c2 in zip(instance1, instance2))
    return distance


def manhattan_distance(instance1, instance2, length):
    distance = sum(abs(num1 - num2) for num1, num2 in zip(instance1, instance2))
    return distance


def classify_point(k, point, dataset, distance_func):
    distances = []

    for data in dataset:
        label = data[-1]
        d = distance_func(point, data[:-1], len(point))
        obj = DistClass(dist=d, tag=label)
        distances.append(obj)

    distances.sort(key=lambda x: x.dist)
    k_nearest = distances[:k]

    labels = [obj.tag for obj in k_nearest]
    result = max(set(labels), key=labels.count)

    return result


# Make an untagged vector, point, and two tagged vectors, data1 and data2
point = [1, 0, 0, "?"]  # (an unknown tag)
data1 = [1, 1, 1, "M"]
data2 = [1, 3, 0, "F"]

# Write code to separate the data (X) from the tag (Y)

# Find out the distance between data1 and point and data2 and point

# Get more data from a file, myFile.csv
url = "https://github.com/rosenfa/ai/blob/master/myFile.csv?raw=true"
df = pd.read_csv(url, header=0, on_bad_lines="skip")
dataset = np.array(df)

# Assume you have a new value for point: [0, 0, 100]
point = [0, 0, 100]
euc_distances = []

for temp in dataset:
    label = temp[-1]
    d = euclidean_distance(point, temp, 3)
    obj = DistClass()
    obj.dist = d
    obj.tag = label
    euc_distances.append(obj)

euc_distances.sort(key=lambda x: x.dist)

# Questions:
# 1. What is the label for point if k=1?
# 2. What is the label for point if k=3?
# 3. Would the result be different if we used a different distance function like Hamming or Manhattan?

# Implement Hamming and Manhattan distances and test for k=1 and k=3 for all possibilities
ham_distances = []
man_distances = []

for temp in dataset:
    label = temp[-1]
    d_ham = hamming_distance(point, temp[:-1], 3)
    d_man = manhattan_distance(point, temp[:-1], 3)

    obj_ham = DistClass()
    obj_ham.dist = d_ham
    obj_ham.tag = label
    ham_distances.append(obj_ham)

    obj_man = DistClass()
    obj_man.dist = d_man
    obj_man.tag = label
    man_distances.append(obj_man)

ham_distances.sort(key=lambda x: x.dist)
man_distances.sort(key=lambda x: x.dist)

# Check the results for Hamming and Manhattan distances


# Now look at bigger files: mytrain.csv (for training the model) and mytest.csv (for testing)
url_train = "https://github.com/rosenfa/ai/blob/master/mytrain.csv?raw=true"
url_test = "https://github.com/rosenfa/ai/blob/master/mytest.csv?raw=true"
train_data = np.array(pd.read_csv(url_train, header=0, on_bad_lines="skip"))
test_data = np.array(pd.read_csv(url_test, header=0, on_bad_lines="skip"))


# Implement the knn code with 3 different values for k and Euclidean distance
k_values = [1, 7, 15]
for k in k_values:
    result_euc = []
    for test_point in test_data:
        label_euc = classify_point(k, test_point[:-1], train_data, euclidean_distance)
        result_euc.append(label_euc)

    accuracy_euc = np.mean(result_euc == test_data[:, -1])
    e_ans.append(accuracy_euc)

# Now see if using Hamming or Manhattan distance gives better results
for k in k_values:
    result_ham = []
    result_man = []

    for test_point in test_data:
        label_ham = classify_point(k, test_point[:-1], train_data, hamming_distance)
        label_man = classify_point(k, test_point[:-1], train_data, manhattan_distance)

        result_ham.append(label_ham)
        result_man.append(label_man)

    accuracy_ham = np.mean(result_ham == test_data[:, -1])
    accuracy_man = np.mean(result_man == test_data[:, -1])

    h_ans.append(accuracy_ham)
    m_ans.append(accuracy_man)

print("|||||")
print("|-|-|-|-|")
print("k", 1, 7, 15, sep="|")
print("euclid", *e_ans, sep="|")
print("hamming", *h_ans, sep="|")
print("manhattan", *m_ans, sep="|")

assert e_ans == [0.5, 0.74, 0.7]
assert h_ans == [0.61, 0.55, 0.57]
assert m_ans == [0.61, 0.63, 0.69]

"""
k,1,7,15
euclid,0.5,0.74,0.7
hamming,0.61,0.55,0.57
manhattan,0.61,0.63,0.69
"""
