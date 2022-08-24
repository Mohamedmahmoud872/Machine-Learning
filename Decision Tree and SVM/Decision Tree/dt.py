import math
import random
import statistics
import numpy as np
import pandas as pd


class Tree:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.label = None
        self.count = None
        self.split_feature_value = None
        self.split_feature = None


def data_to_distribution(data):
    Ys = []
    for (point, y) in data:
        Ys.append(y)
    num_of_entries = len(Ys)
    unique_ys = set(Ys)  # get the unique values of Ys
    distribution = [(float(Ys.count(y)) / num_of_entries) for y in unique_ys]
    return distribution


def split_data(data, feature_index):
    attribute_values = []
    for (point, y) in data:
        attribute_values.append(point[feature_index])

    unique_attribute_values = set(attribute_values)
    for value in unique_attribute_values:
        subset = []
        for (point, y) in data:
            if point[feature_index] == value:
                subset.append((point, y))
        yield subset


def split_to_train_test(data, train_size):
    if isinstance(train_size, float):
        train_size = round(train_size * len(data))
    idxs = data.index.tolist()
    train_indices = random.sample(population=idxs, k=train_size)
    train = data.loc[train_indices]
    test = data.drop(train_indices)
    return train, test


def replace_missing_data(data):
    for col in data.columns:
        expected = statistics.mode(data[col])
        data[col] = np.where((data[col].values == '?'), expected, data[col].values)
    return data


def majority(data, node):
    labels = [label for (pt, label) in data]
    choice = max(set(labels), key=labels.count)
    node.label = choice
    node.count = dict([(label, labels.count(label)) for label in set(labels)])

    return node


def build_tree(data, root, left):
    unique_values = []
    for (point, y) in data:
        unique_values.append(y)
    unique_values = set(unique_values)
    check_label = True if len(unique_values) <= 1 else False

    if check_label:
        root.label = data[0][1]
        root.count = {root.label: len(data)}
        return root

    if len(left) == 0:  # length of remaining features
        return majority(data, root)

    optimal = max(left, key=lambda index: information_gain(data, index))
    if information_gain(data, optimal) == 0: return majority(data, root)

    root.split_feature = optimal
    for dataSubset in split_data(data, optimal):
        child = Tree(parent=root)
        child.split_feature_value = dataSubset[0][0][optimal]
        root.children.append(child)
        build_tree(dataSubset, child, left - set([optimal]))
    return root


def entropy(data):
    e = 0
    values = []
    for i in data:
        values.append(i * np.log2(i))
    e = -sum(values)
    return e


def information_gain(data, feature_index):
    entropy_gain = entropy(data_to_distribution(data))
    data_subset = split_data(data, feature_index)
    for i in data_subset: entropy_gain -= entropy(data_to_distribution(i))
    return entropy_gain


def predict(tree, point):
    if tree.children == []:
        return tree.label
    else:
        values = []
        for i in tree.children:
            if i.split_feature_value == point[tree.split_feature]:
                values.append(i)
        return predict(values[0], point)


def calculate_accuracy(test, predicted):
    accuracy = 0
    labels = []
    for (point, label) in test:
        labels.append(label)
    fixed_labels = []
    for (a, b) in zip(labels, predicted):
        if a == b:
            fixed_labels.append(1)
        else:
            fixed_labels.append(0)
    accuracy = sum(fixed_labels) / len(labels)
    return accuracy

def testing(df, ratios):
    iterations = len(ratios)
    for size in ratios:
        print("For training ratio: ", size)
        accuracy = []
        for j in range(iterations):
            prepared_data = replace_missing_data(df)
            train, test = split_to_train_test(prepared_data, size)
            train_array = train.to_numpy()
            np.savetxt("train_file.txt", train_array, fmt="%s")
            test_array = test.to_numpy()
            np.savetxt("test_file.txt", test_array, fmt="%s")

            with open('train_file.txt', 'r') as train_file:
                train_file_by_lines = train_file.readlines()
            train_data = [line.strip().split(' ') for line in train_file_by_lines]
            train_data = [(x[1:], x[0]) for x in train_data]

            tree = build_tree(train_data, Tree(), set(range(len(train_data[0][0]))))

            with open('test_file.txt', 'r') as testFileContent:
                test_file_by_lines = testFileContent.readlines()
            test_data = [line.strip().split(' ') for line in test_file_by_lines]
            test_data = [(x[1:], x[0]) for x in test_data]
            predicted_values = []
            for i in test_data:
                predicted_value = [predict(tree, issues) for issues, political in test_data]
            accuracy.append(calculate_accuracy(test_data, predicted_value))

        print("Max accuracy: ", max(accuracy) * 100)
        print("Min accuracy: ", min(accuracy) * 100)
        print("Mean accuracy: ", (sum(accuracy) / len(accuracy)) * 100)
        print("=================================================================")



if __name__ == '__main__':
    df = pd.read_csv("house-votes-84.data.txt", header=None)
    ratios = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    testing(df, ratios)

