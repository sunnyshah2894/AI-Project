#!/usr/bin/env python
import sys
import csv
from collections import Counter


import argparse

import numpy as np


spam_words = Counter()
ham_words = Counter()
count_spam_words = 0
count_ham_words = 0
count_spam = 0
count_ham = 0


def train(train_file):
    global spam_words, ham_words, count_ham_words, count_spam_words, count_spam, count_ham
    train_data = open(train_file, "r")

    for data in train_data.readlines():

        row = data.split(' ')
        actual_label = row[1]
        for i in range(2, len(row), 2):
            word = row[i]
            count = row[i+1]
            if actual_label.lower() == 'spam':
                spam_words[word] += int(count)
                count_spam += 1
            else:
                ham_words[word] += int(count)
                count_ham += 1

    for word in spam_words:
        count_spam_words += spam_words[word]
    for word in ham_words:
        count_ham_words += ham_words[word]

def test(test_file):

    test_data = open(test_file, "r")
    spam_probability = float(count_spam) /float(count_ham + count_spam)
    ham_probability = float(count_ham) / float(count_ham + count_spam)
    delta = 0.1
    spam_vocab = len(spam_words)
    ham_vocab=len(spam_words)
    predictions = []

    for row in test_data.readlines():

        row = row.split(' ')
        actual_label = row[1]
        # Due to probabilities in logarithms, prob. of spam or ham is also taken in logarithms form
        probability_of_being_spam = float(np.log10(spam_probability))
        # now add conditional probability
        for i in range(2, len(row), 2):
            word = row[i]
            count = row[i+1]
            numerator = float( spam_words[word] + delta )
            denominator = float( count_spam_words + ( spam_vocab*delta ))
            probability_of_being_spam += np.log10(numerator/denominator) * float(count)

        probability_of_being_ham = np.log10(ham_probability)

        # Now add conditional probability
        for i in range(2, len(row), 2):
            word = row[i]
            count = row[i+1]
            numerator = float( ham_words[word] + delta )
            denominator = float( count_ham_words + ( ham_vocab*delta ))
            probability_of_being_ham += np.log10(numerator/denominator) * float(count)

        predicted_label = "spam"
        if( probability_of_being_ham > probability_of_being_spam ):
            predicted_label = "ham"

        predictions.append((row[0], predicted_label,actual_label))


    return predictions


def load_input():
    global train_data, test_data, output
    parser=argparse.ArgumentParser()
    parser.add_argument('-f1', required=True, help='train data', dest='train_data')
    parser.add_argument('-f2', required=True, help='test data', dest="test_data")
    parser.add_argument('-o', required=True, help='output', dest="output")
    inputs=parser.parse_args()
    return inputs.train_data, inputs.test_data, inputs.output


def evaluate(predictions, output_file):

    number_of_correct_spam_predictions=0
    number_of_correct_ham_predictions=0
    total_test_instances=len(predictions)
    true_ham_count=0
    true_spam_count=0
    predicted_spam_count=0
    predicted_ham_count=0

    output=output_file + ".csv"

    with open(output, 'wb') as writer:

        for prediction in predictions:
            email_id = prediction[0]
            predicted_result = prediction[1]
            actual_result = prediction[2]

            if actual_result == predicted_result:
                if predicted_result == "spam":
                    number_of_correct_spam_predictions += 1
                else:
                    number_of_correct_ham_predictions += 1

            if actual_result.lower() == "spam":
                true_spam_count+=1
            else:
                true_ham_count+=1

            if predicted_result.lower() == "spam":
                predicted_spam_count+=1
            else:
                predicted_ham_count+=1

            writer.write(email_id+","+predicted_result+"\n")

    true_positive = number_of_correct_spam_predictions
    false_positive = predicted_spam_count - number_of_correct_spam_predictions
    false_negative = predicted_ham_count - number_of_correct_ham_predictions

    precision = float(float(true_positive) / float(true_positive+false_positive)) * 100
    recall = float(float(true_positive) / float(true_positive + false_negative)) * 100
    F1Score = 2.0*(precision*recall) / (precision + recall)

    print "Accuracy: ", float(number_of_correct_ham_predictions + number_of_correct_spam_predictions) / (total_test_instances) * 100, "%"
    print "Precision: ", precision, "%"
    print "Recall: ", recall, "%"
    print "F1-Score: ", F1Score,"%"

if __name__ == "__main__":

    train_data, test_data, output = load_input()

    train(train_data)
    predictions = test(test_data)

    evaluate(predictions, output)