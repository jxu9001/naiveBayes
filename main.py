# Jerry Xu
# CS 4375 Fall 2021 Homework 2 Part 2 (Naive Bayes Classifier)

import collections
import itertools
import math
import random
import sys


def create_table(dataset, attributes):
    """
    creates a lookup table which can then be used for naive bayes classification
    """
    lookup_table = collections.defaultdict(int)

    for example in dataset:
        # get the count of examples in each class
        lookup_table["class={}".format(example["class"])] += 1

        for attribute, value in itertools.product(attributes[:-1], [0, 1]):
            # get the count of examples w/ each attribute value in class 0
            # e.g. count of examples where wesley=0 | class=0, wesley=1 | class=0, etc.
            if example["class"] == 0 and example[attribute] == value:
                lookup_table["{}={}|0".format(attribute, value)] += 1

            # get the count of examples w/ each attribute value in class 1
            # e.g. count of examples where wesley=0 | class=1, wesley=1 | class=1, etc.
            elif example["class"] == 1 and example[attribute] == value:
                lookup_table["{}={}|1".format(attribute, value)] += 1

    return lookup_table


def print_probabilities(lookup_table, attributes):
    """
    prints the probabilities associated with each class
    """
    num_c0 = lookup_table["class=0"]
    num_c1 = lookup_table["class=1"]
    probs_c0 = ["P(class=0)={:.2f} ".format(num_c0 / (num_c0 + num_c1))]
    probs_c1 = ["P(class=1)={:.2f} ".format(num_c1 / (num_c0 + num_c1))]

    for attribute in attributes[:-1]:
        for val in "01":
            key0 = "{}={}|0".format(attribute, val)
            key1 = "{}={}|1".format(attribute, val)
            prob0 = lookup_table[key0] / num_c0 if num_c0 != 0 else 0
            prob1 = lookup_table[key1] / num_c1 if num_c1 != 0 else 0
            probs_c0.append("P({})={:.2f} ".format(key0, prob0))
            probs_c1.append("P({})={:.2f} ".format(key1, prob1))

    print("".join(probs_c0))
    print("".join(probs_c1))


def predict(example, lookup_table):
    """
    returns the classifier's prediction when given an example
    """
    num_c0 = lookup_table["class=0"]
    num_c1 = lookup_table["class=1"]
    log_prob_c0 = math.log(num_c0 / (num_c0 + num_c1))
    log_prob_c1 = math.log(num_c1 / (num_c0 + num_c1))
    keys0 = ["{}={}|0".format(k, v) for k, v in example.items() if k != "class"]
    keys1 = ["{}={}|1".format(k, v) for k, v in example.items() if k != "class"]

    for key0, key1 in zip(keys0, keys1):
        prob0 = lookup_table[key0] / num_c0
        prob1 = lookup_table[key1] / num_c1

        if num_c0 == 0 or prob0 == 0:
            continue
        log_prob_c0 += math.log(prob0)

        if num_c1 == 0 or prob1 == 0:
            continue
        log_prob_c1 += math.log(prob1)

    # tiebreaker scenarios if both predicted classes are equally likely
    if log_prob_c0 == log_prob_c1:
        if num_c0 > num_c1:
            return 0
        elif num_c0 < num_c1:
            return 1
        else:
            return random.randint(0, 1)
    else:
        return [log_prob_c0, log_prob_c1].index(max([log_prob_c0, log_prob_c1]))


def accuracy(lookup_table, dataset):
    """
    returns the accuracy of the decision tree when tested on a dataset
    """
    correct = 0

    for example in dataset:
        if predict(example, lookup_table) == example["class"]:
            correct += 1

    return correct / len(dataset)


def main():
    # command line args
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]

    # read the training and test sets from their respective files
    with open(train_file_name) as f:
        # some python magic to skip over all empty lines
        train_dataset = [line.split() for line in f if line.strip()]

    attributes = train_dataset.pop(0)

    # more python magic to get the dataset into a usable format
    # now the dataset is a list of dictionaries in the following form: {'a0': 1, 'a1': 1, 'a2': 0, 'a3': 0, 'class': 1}
    train_dataset = [dict(zip(attributes, list(map(int, example)))) for example in train_dataset]

    with open(test_file_name) as f:
        test_dataset = [line.split() for line in f if line.strip()]

    test_dataset.pop(0)
    test_dataset = [dict(zip(attributes, list(map(int, example)))) for example in test_dataset]

    # Part A: Train the naive bayes classifier on the training set and print the class probabilities to screen
    lookup_table = create_table(train_dataset, attributes)
    print_probabilities(lookup_table, attributes)

    # Part B: Test the naive bayes classifier on the training set and print the accuracy to screen
    train_accuracy = accuracy(lookup_table, train_dataset)
    print("\nAccuracy on training set ({} instances): {:.2f}%".format(len(train_dataset), 100 * train_accuracy))

    # Part C: Test the naive bayes classifier on the test set and print the accuracy to screen
    test_accuracy = accuracy(lookup_table, test_dataset)
    print("\nAccuracy on test set ({} instances): {:.2f}%".format(len(test_dataset), 100 * test_accuracy))


if __name__ == "__main__":
    main()
