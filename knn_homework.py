#!.e/bin/python3.12
# -*- coding: utf-8 -*-

import math
import pandas as pd
import numpy as np

e_ans = []  # List to store accuracy results for Euclidean distance
h_ans = []  # List to store accuracy results for Hamming distance
m_ans = []  # List to store accuracy results for Manhattan distance


class Clazz:
    def __init__(self, dist=-1, tag="-"):
        self.dist = dist
        self.tag = tag


def euclidean_distance(instance1, instance2, length):
    # Calculate Euclidean distance between two instances
    distance = 0
    for x in range(length):
        num1 = float(instance1[x])
        num2 = float(instance2[x])
        distance += pow(num1 - num2, 2)
    return math.sqrt(distance)


def hamming_distance(instance1, instance2, length):
    # Calculate Hamming distance between two instances
    distance = sum(c1 != c2 for c1, c2 in zip(instance1, instance2))
    return distance


def manhattan_distance(instance1, instance2, length):
    # Calculate Manhattan distance between two instances
    distance = sum(abs(num1 - num2) for num1, num2 in zip(instance1, instance2))
    return distance


def classify_point(k, point, dataset, distance_func):
    """
    Classify a point based on k-nearest neighbors using the specified distance function.

    Parameters:
    - k (int): Number of neighbors to consider.
    - point (list): The point to be classified.
    - dataset (list): The dataset containing labeled points.
    - distance_func (function): The distance function to calculate distances between points.

    Returns:
    - str: The predicted label for the given point.
    """

    # Step 1: Calculate distances between the given point and all points in the dataset
    distances = []

    for data in dataset:
        # Extract the label and calculate the distance
        label = data[-1]
        d = distance_func(point, data[:-1], len(point))

        # Create an object to store the distance and label
        obj = Clazz(dist=d, tag=label)
        distances.append(obj)

    # Step 2: Sort the distances in ascending order
    distances.sort(key=lambda x: x.dist)

    # Step 3: Select the k-nearest neighbors
    k_nearest = distances[:k]

    # Step 4: Extract labels from the k-nearest neighbors
    labels = [obj.tag for obj in k_nearest]

    # Step 5: Determine the most common label among the k-nearest neighbors
    result = max(set(labels), key=labels.count)

    # Step 6: Return the predicted label
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
    obj = Clazz()
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

# Loop through each data point in the dataset
for temp in dataset:
    # Extract the label of the current data point
    label = temp[-1]

    # Calculate Hamming distance between the given point and the current data point
    d_ham = hamming_distance(point, temp[:-1], 3)

    # Calculate Manhattan distance between the given point and the current data point
    d_man = manhattan_distance(point, temp[:-1], 3)

    # Create an object to store Hamming distance, label, and tag
    obj_ham = Clazz()
    obj_ham.dist = d_ham
    obj_ham.tag = label

    # Append the object to the list of Hamming distances
    ham_distances.append(obj_ham)

    # Create an object to store Manhattan distance, label, and tag
    obj_man = Clazz()
    obj_man.dist = d_man
    obj_man.tag = label

    # Append the object to the list of Manhattan distances
    man_distances.append(obj_man)

# Sort the lists of Hamming and Manhattan distances based on the distance value
ham_distances.sort(key=lambda x: x.dist)
man_distances.sort(key=lambda x: x.dist)

# Check the results for Hamming and Manhattan distances


# Now look at bigger files: mytrain.csv (for training the model) and mytest.csv (for testing)
url_train = "https://github.com/rosenfa/ai/blob/master/mytrain.csv?raw=true"
url_test = "https://github.com/rosenfa/ai/blob/master/mytest.csv?raw=true"
train_data = np.array(pd.read_csv(url_train, header=0, on_bad_lines="skip"))
test_data = np.array(pd.read_csv(url_test, header=0, on_bad_lines="skip"))


# Implement the k-NN code with 3 different values for k and Euclidean distance
k_values = [1, 7, 15]

# Loop over each k value
for k in k_values:
    # Initialize an empty list to store the results for Euclidean distance
    result_euc = []

    # Loop over each test data point
    for test_point in test_data:
        # Classify the test point using k-NN with Euclidean distance
        label_euc = classify_point(k, test_point[:-1], train_data, euclidean_distance)
        result_euc.append(label_euc)

    # Calculate the accuracy for Euclidean distance and append to the result list
    accuracy_euc = np.mean(result_euc == test_data[:, -1])
    e_ans.append(accuracy_euc)

# Now see if using Hamming or Manhattan distance gives better results

# Loop over each k value
for k in k_values:
    # Initialize empty lists to store the results for Hamming and Manhattan distances
    result_ham = []
    result_man = []

    # Loop over each test data point
    for test_point in test_data:
        # Classify the test point using k-NN with Hamming distance
        label_ham = classify_point(k, test_point[:-1], train_data, hamming_distance)
        result_ham.append(label_ham)

        # Classify the test point using k-NN with Manhattan distance
        label_man = classify_point(k, test_point[:-1], train_data, manhattan_distance)
        result_man.append(label_man)

    # Calculate the accuracy for Hamming and Manhattan distances and append to the result lists
    accuracy_ham = np.mean(result_ham == test_data[:, -1])
    accuracy_man = np.mean(result_man == test_data[:, -1])

    # Append the accuracies to the respective lists
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

"""
1. What is the label for point if k=1?
The label for the point when k=1 using Euclidean distance is obtained from the nearest neighbor, so you would need to check the label of the closest data point to the given point.
In this case, you can look at the label of the closest point in euc_distances.

2. What is the label for point if k=3?
For k=3, you need to consider the labels of the three nearest neighbors and choose the most common label among them.
This information is stored in euc_distances.
The result for k=3 is obtained by checking the most frequent label among the three closest neighbors.

3. Would the result be different if we used a different distance function like Hamming or Manhattan?
Yes, the result would likely be different if you use a different distance function.
The choice of distance function affects how the algorithm measures the similarity between data points.
Different distance functions are suitable for different types of data.
For example, Euclidean distance is suitable for continuous numerical data, while Hamming distance is suitable for categorical data with a binary representation.
Manhattan distance is another option for numerical data, but it considers the sum of absolute differences instead of squared differences.
"""
