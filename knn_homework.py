#!.e/bin/python3.12
# -*- coding: utf-8 -*-

import math
import pandas as pd
import numpy as np

class DistClass:
    def __init__(self, dist=-1, tag='-'):
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
point = [1, 0, 0, '?']  # (an unknown tag)
data1 = [1, 1, 1, 'M']
data2 = [1, 3, 0, 'F']

# Write code to separate the data (X) from the tag (Y)
print("The vector ", data1[0:-1], " has tag ", data1[-1])

# Find out the distance between data1 and point and data2 and point
print("Euclidean distance between data1 and point:", euclidean_distance(data1, point, 3))

# Get more data from a file, myFile.csv
url = 'https://github.com/rosenfa/ai/blob/master/myFile.csv?raw=true'
df = pd.read_csv(url, header=0, on_bad_lines='skip')
dataset = np.array(df)

# Print the first two vectors in the file
print(dataset[:2])

# Print the Euclidean distance between those two vectors
print("Euclidean distance between the first two vectors:", euclidean_distance(dataset[0], dataset[1], 3))

# Assume you have a new value for point: [0, 0, 100]
point = [0, 0, 100]
euc_distances = []

for temp in dataset:
    label = temp[-1]
    d = euclidean_distance(point, temp, 3)
    print("The distance between ", point, " and ", temp, " is ", str(d))
    print("And the label is " + label)
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

    print("Hamming distance between ", point, " and ", temp, " is ", str(d_ham))
    print("Manhattan distance between ", point, " and ", temp, " is ", str(d_man))

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
print("Hamming distances:")
for obj in ham_distances:
    print("Distance: ", obj.dist, ", Tag: ", obj.tag)

print("\nManhattan distances:")
for obj in man_distances:
    print("Distance: ", obj.dist, ", Tag: ", obj.tag)

# Now look at bigger files: mytrain.csv (for training the model) and mytest.csv (for testing)
url_train = 'https://github.com/rosenfa/ai/blob/master/mytrain.csv?raw=true'
url_test = 'https://github.com/rosenfa/ai/blob/master/mytest.csv?raw=true'
train_data = np.array(pd.read_csv(url_train, header=0, on_bad_lines='skip'))
test_data = np.array(pd.read_csv(url_test, header=0, on_bad_lines='skip'))

print("\nTraining data shape:", train_data.shape)
print("Training data:")
print(train_data)

print("\nTest data shape:", test_data.shape)
print("Test data:")
print(test_data)

# Implement the knn code with 3 different values for k and Euclidean distance
k_values = [1, 7, 15]
for k in k_values:
    result_euc = []
    for test_point in test_data:
        label_euc = classify_point(k, test_point[:-1], train_data, euclidean_distance)
        result_euc.append(label_euc)

    accuracy_euc = np.mean(result_euc == test_data[:, -1])
    print(f"\nK = {k} with Euclidean distance:")
    print("Predicted labels:", result_euc)
    print("Accuracy:", accuracy_euc)

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

    print(f"\nK = {k} with Hamming distance:")
    print("Predicted labels:", result_ham)
    print("Accuracy:", accuracy_ham)

    print(f"\nK = {k} with Manhattan distance:")
    print("Predicted labels:", result_man)
    print("Accuracy:", accuracy_man)
