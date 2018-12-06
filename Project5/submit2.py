import pickle as pkl
import argparse
import csv
import numpy as np
import pandas as pd

from scipy.stats import chisquare
import sys

sys.setrecursionlimit(4000)

'''
TreeNode represents a node in your decision tree
TreeNode can be:
	- A non-leaf node: 
		- data: contains the feature number this node is using to split the data
		- children[0]-children[4]: Each correspond to one of the values that the feature can take

	- A leaf node:
		- data: 'T' or 'F' 
		- children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.
'''


# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes=list(children)
        self.data=data

    def save_tree(self, filename):
        obj=open(filename, 'w')
        pkl.dump(self, obj)

def count_of_value_in_a_column( df, column_name, value ):
    return len(df[df[column_name] == value])

def filter_dataframe_rowwise( df, column_name, value ):
    return df[df[column_name] == value]

def generate_decision_tree(examples, attributes, p_value):

    if examples.shape[0] == 0:
        root = TreeNode(data='T', children=[])
        return root

    if len(examples['target'].unique()) == 1:
        root=TreeNode(data='T', children=[])
        if examples['target'].unique()[0] == 0:
            root.data = 'F'
        return root

    current_count_of_positives = count_of_value_in_a_column( examples,'target',1 )
    current_count_of_negatives = count_of_value_in_a_column( examples,'target',0 )

    if len(attributes) == 0:
        root = TreeNode(data='T', children=[])
        if current_count_of_negatives > current_count_of_positives:
            root.data = 'F'
        return root


    # selecting attribute with least entropy to split on
    best_attribute_till_now = find_best_attribute_to_split_on(examples, attributes)

    # using chi-squared criterion to decide whether to stop
    should_we_stop_splitting = chi2_splitting(examples, best_attribute_till_now, p_value)

    # if p-value is greater than threshold
    if should_we_stop_splitting == True:
        root=TreeNode(data='T', children=[])
        if current_count_of_negatives > current_count_of_positives:
            root.data='F'
        return root

    root = TreeNode( data=str(best_attribute_till_now+1) )

    new_attributes = []
    for i in attributes:
        if i != best_attribute_till_now:
            new_attributes.append(i)

    for value in range(0,5):
        new_examples = filter_dataframe_rowwise( examples,best_attribute_till_now,value+1 )
        new_examples = new_examples.drop([best_attribute_till_now], axis=1)
        root.nodes[value] = generate_decision_tree(new_examples, new_attributes, p_value)

    return root

def calculate_entropy( q ):
    if( q==0 or q==1 ):
        return 0.0
    return (( q*np.log2(q) + (1-q) * np.log2(1-q) ) * -1.0)

# this function returns the attribute with maximum gain (least entropy)
def find_best_attribute_to_split_on(examples, attr):

    attribute_entropies = {}
    p_plus_n = examples.shape[0]

    for attribute in attr:
        current_entropy = 0.0

        for value in examples[attribute].unique():

            filtered_data = filter_dataframe_rowwise(examples,attribute,value)

            p_k = count_of_value_in_a_column( filtered_data,'target',1 )
            n_k = count_of_value_in_a_column( filtered_data,'target',0 )

            current_entropy += (float(p_k+n_k)/float(p_plus_n))*calculate_entropy(p_k/(p_k+n_k))

        attribute_entropies[attribute] = current_entropy

    # Maximum information gain can be evaluated using the attribute with minimum entropy sum
    attributes_sorted_as_per_entropy = sorted( attribute_entropies.items() , key=lambda k:k[1])

    return attributes_sorted_as_per_entropy[0][0]


def chi2_splitting(examples, best_attr, p_value):

    p = count_of_value_in_a_column( examples,'target',1 )
    n = count_of_value_in_a_column( examples,'target',0 )
    S = 0.0
    list_of_feature_values = examples[best_attr].unique()

    observed_positives_and_negatives = []
    expected_positives_and_negatives = []

    for elem in list_of_feature_values:

        filtered_data = filter_dataframe_rowwise(examples, best_attr, elem)

        T_i = filtered_data.shape[0]
        p_i = count_of_value_in_a_column( filtered_data,'target',1 )
        n_i = count_of_value_in_a_column( filtered_data,'target',0 )

        p_i_prime = p * T_i / float( p+n )
        n_i_prime = n * T_i / float( p+n )

        if p_i_prime != 0:
            expected_positives_and_negatives.append(p_i_prime)
            observed_positives_and_negatives.append(p_i)
        if n_i_prime != 0:
            expected_positives_and_negatives.append(n_i_prime)
            observed_positives_and_negatives.append(n_i)

    _, p = chisquare(observed_positives_and_negatives, expected_positives_and_negatives)

    if p > p_value:
        return True
    else:
        return False

def evaluate_datapoint(root, sample):
    if root.data == 'T':
        return 1
    if root.data == 'F':
        return 0
    column_id_of_root = int(root.data) - 1
    column_value_in_given_sample = sample[column_id_of_root] - 1
    return evaluate_datapoint(root.nodes[column_value_in_given_sample], sample)

# loads Train and Test data
def load_data(ftrain, ftest):
	Xtrain, Ytrain, Xtest = [],[],[]
	with open(ftrain, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtrain.append(rw)

	with open(ftest, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtest.append(rw)

	ftrain_label = ftrain.split('.')[0] + '_label.csv'
	with open(ftrain_label, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = int(row[0])
	        Ytrain.append(rw)

	print('Data Loading: done')
	return Xtrain, Ytrain, Xtest


parser=argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args=vars(parser.parse_args())

p_value = float(args['p'])
Xtrain_name=args['f1']
Ytrain_name=args['f1'].split('.')[0] + '_labels.csv'  # labels filename will be the same as training file name but with _label at the end

Xtest_name=args['f2']
Ytest_predict_name=args['o']

tree_name=args['t']

Xtrain, Ytrain, Xtest=load_data(Xtrain_name, Xtest_name)

print("Training...")
Xtrain_df = pd.DataFrame(Xtrain)
Ytrain_df = pd.DataFrame(Ytrain)

attributes = [i for i in range(Xtrain_df.shape[1])]

Xtrain_df['target'] = Ytrain_df[0]


model = generate_decision_tree(Xtrain_df, attributes, p_value)


model.save_tree(tree_name)

print("Testing...")
Ypredict=[]
for i in range(0, len(Xtest)):
    result=evaluate_datapoint(model, Xtest[i])
    Ypredict.append([result])

with open(Ytest_predict_name, "wb") as f:
    writer=csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")
