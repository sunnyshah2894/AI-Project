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
    """
        Function to find the count of occurrence of a particular "value" in column "column_name" of dataframe "df"
    """
    return len(df[df[column_name] == value])

def filter_dataframe_rowwise( df, column_name, value ):
    """
        Function to filter the dataframe along the column "column_name" with value of "value"
    """
    return df[df[column_name] == value]

def generate_decision_tree(examples, attributes, p_value):

    """
        What if there are no examples left, we create a terminal node and return
    """
    if examples.shape[0] == 0:
        next_node = TreeNode(data='T', children=[])
        return next_node

    """
        What if the target column has only either positives or negatives left
    """
    if len(examples['target'].unique()) == 1:
        next_node=TreeNode(data='T', children=[])
        if examples['target'].unique()[0] == 0:
            next_node.data = 'F'
        return next_node

    """
        Find the count of the positive(1)/negatives(0) present in the current dataset i.e. examples
    """
    current_count_of_positives = count_of_value_in_a_column( examples,'target',1 )
    current_count_of_negatives = count_of_value_in_a_column( examples,'target',0 )


    """
        If no attributes left to break down further, create a terminal node with plurarity value of the either positive
        or negative depending on the count of positive and negatives remaining in the dataset.
    """
    if len(attributes) == 0:
        next_node = TreeNode(data='T', children=[])
        if current_count_of_negatives > current_count_of_positives:
            next_node.data = 'F'
        return next_node

    """
        Find the best attribute, i.e. the attribute that is not yet selected, but upon selection gives maximum information gain
    """
    best_attribute_till_now = find_best_attribute_to_split_on(examples, attributes)

    """
        Check if we should stop splitting, based on the chisqare statistics
    """
    should_we_stop_splitting = check_chi_square_stopping_condition(examples, best_attribute_till_now, p_value)

    """
        If we should stop now, then we create a terminal node with plurarity value of the either positive
        or negative depending on the count of positive and negatives remaining in the dataset.
    """
    if should_we_stop_splitting == True:
        next_node = TreeNode(data='T', children=[])
        if current_count_of_negatives > current_count_of_positives:
            next_node.data='F'
        return next_node
    else:
        """
            Add the best attribute node to the tree and recursively call to create the tree for its feature values   
        """
        next_node = TreeNode( data=str(best_attribute_till_now+1) )

        """
            Create a list of new attributes i.e. remove the current attribute from the current attribute list
        """
        new_attributes = []
        for i in attributes:
            if i != best_attribute_till_now:
                new_attributes.append(i)

        """
            Recursively call to create the tree for its feature values i.e. in our case 5 bin values
        """
        for value in range(0,5):
            new_examples = filter_dataframe_rowwise( examples,best_attribute_till_now,value+1 )
            new_examples = new_examples.drop([best_attribute_till_now], axis=1)
            next_node.nodes[value] = generate_decision_tree(new_examples, new_attributes, p_value)

        return next_node


def calculate_entropy( q ):
    """
        Calculates the entropy B(q) given by B(q) = -q(log2(q)) - (1-q)log2(1-q)
    """
    if( q==0 or q==1 ):
        return 0.0
    return (( q*np.log2(q) + (1-q) * np.log2(1-q) ) * -1.0)


def find_best_attribute_to_split_on(examples, attr):

    """
        Find the best attribute to split the currently avaliable dataset.
        The best attribute is the one that gives maximum information gain i.e. the one that gives minimum entropy

        information_gain of attribute = B(p/(p+n)) - Sum of entropy of each element of attribute

    """

    entropies_for_each_attribute = {}
    p_plus_n = examples.shape[0]

    for attribute in attr:

        current_entropy = 0.0
        for value in examples[attribute].unique():

            filtered_data = filter_dataframe_rowwise(examples,attribute,value)

            p_k = count_of_value_in_a_column( filtered_data,'target',1 )
            n_k = count_of_value_in_a_column( filtered_data,'target',0 )

            # Sum of entropy of each element of attribute
            current_entropy += (float(p_k+n_k)/float(p_plus_n))*calculate_entropy(p_k/(p_k+n_k))

        entropies_for_each_attribute[attribute] = current_entropy

    # Maximum information gain can be evaluated using the attribute with minimum entropy sum
    attributes_sorted_as_per_entropy = sorted( entropies_for_each_attribute.items() , key=lambda k:k[1])

    # Return the index of the attribute for which we got minimum entropy
    return attributes_sorted_as_per_entropy[0][0]


def check_chi_square_stopping_condition(examples, best_attr, p_value):

    """
        Let p, n denote the number of positive and negative examples that we have in our
        dataset (not the whole set, the remaining one that we work on at this node).
        Let (N is the total number of examples in the current dataset):
            p_i_prime = p*(T_i)/N
            n_i_prime = n*(T_i)/N
        be the expected number of positives and negatives in each partition, if the attribute is irrelevant to the class.
        Then the statistic of interest is the chi-square quantity where p_i, n_i are the positives and negatives in partition T_i.
    """

    """
        Find the count of the positive(1)/negatives(0) present in the current dataset i.e. examples
    """
    p = count_of_value_in_a_column( examples,'target',1 )
    n = count_of_value_in_a_column( examples,'target',0 )

    list_of_feature_values = examples[best_attr].unique()

    """
        We create 2 lists:
        observed_positives_and_negatives -> stores the p_i and n_i for each T_i
        expected_positives_and_negatives -> stores the p_i_prime and n_i_prime for each T_i
    """
    observed_positives_and_negatives = []
    expected_positives_and_negatives = []

    for elem in list_of_feature_values:

        filtered_data = filter_dataframe_rowwise(examples, best_attr, elem)

        T_i = filtered_data.shape[0]
        p_i = count_of_value_in_a_column( filtered_data,'target',1 )
        n_i = count_of_value_in_a_column( filtered_data,'target',0 )

        """
            p_i_prime=p * (T_i) / N
            n_i_prime=n * (T_i) / N
        """
        p_i_prime = p * T_i / float( p+n )
        n_i_prime = n * T_i / float( p+n )

        if p_i_prime != 0:
            expected_positives_and_negatives.append(p_i_prime)
            observed_positives_and_negatives.append(p_i)
        if n_i_prime != 0:
            expected_positives_and_negatives.append(n_i_prime)
            observed_positives_and_negatives.append(n_i)

    # Calculate the chisquare using the scipy.stats.chisquare library function call
    _, p = chisquare(observed_positives_and_negatives, expected_positives_and_negatives)

    # If the Threshold is reached, then stop splitting further, else return false
    if p > p_value:
        return True
    else:
        return False

def evaluate_sample(root, sample):
    """
        If the root is a terminal node, then return the actual truth value associated with it
    """
    if root.data == 'T':
        return 1
    if root.data == 'F':
        return 0

    """
        If the root node is a intermediate node, then branch along the correct path by checking the value of the
        sample data for the feature represented by the root node
    """

    # feature represented by the root node
    column_id_of_root = int(root.data) - 1

    # value of the feature in the sample
    column_value_in_given_sample = sample[column_id_of_root] - 1

    # recursively traverse down the tree
    return evaluate_sample(root.nodes[column_value_in_given_sample], sample)

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

# Generate the initial list of valid attributes on which we can split
attributes = [i for i in range(Xtrain_df.shape[1])]

# Add the labels as a target column in our dataframe
Xtrain_df['target'] = Ytrain_df[0]

# Call ID3 algorithm to create the decision tree
model = generate_decision_tree(Xtrain_df, attributes, p_value)

model.save_tree(tree_name)

print("Testing...")
Ypredict = []
for i in range(0, len(Xtest)):
    result = evaluate_sample(model, Xtest[i])
    Ypredict.append([result])

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")
