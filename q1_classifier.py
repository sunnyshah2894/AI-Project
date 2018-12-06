import random
import argparse, os, sys
import pandas as pd
import numpy as np
import copy
import time
import csv
import pickle as pkl
from scipy.stats import chisquare

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

sys.setrecursionlimit(100000)
# internal nodes count
internal_nodes=0
# leaf nodes count
leaves=0


# evaluate and return the chi square p value
def chisquare_criterion(sample_space, chosen_attribute):
    observed=[]
    expected=[]
    # number of negative samples
    n=(sample_space['output_value'] == 0).sum()
    # number of positive samples
    p=(sample_space['output_value'] == 1).sum()
    N=n + p
    r1=float(p) / N
    r2=float(n) / N
    unique_values=sample_space[chosen_attribute].unique()
    # for each value calculate the expected and observed number of positives and ngeatives
    for value in unique_values:
        attribute_sample_space=sample_space.filter([chosen_attribute, 'output_value'], axis=1)
        attribute_sample_space=attribute_sample_space.loc[(attribute_sample_space[chosen_attribute] == value)]
        T_i=attribute_sample_space['output_value'].count()
        p_prime_i=float(r1) * T_i
        n_prime_i=float(r2) * T_i
        p_i=float((attribute_sample_space['output_value'] == 1).sum())
        n_i=float((attribute_sample_space['output_value'] == 0).sum())
        if p_prime_i != 0:
            expected.append(p_prime_i)
            observed.append(p_i)
        if n_prime_i != 0:
            expected.append(n_prime_i)
            observed.append(n_i)
    # calculate the chi-square value
    c, p=chisquare(observed, expected)
    # return the p value
    return p


class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes=list(children)
        self.data=data

    def save_tree(self, filename):
        obj=open(filename, 'w')
        pkl.dump(self, obj)


# get the best attribute from the remaining attributes
def find_next_attribute(sample_space, attributes):
    min_entropy=None
    best_attribute=None
    # for each attribute, calculate the gain and the attribute with maximum gain
    for attribute in attributes:
        rows_count=sample_space[attribute].count()
        unique_values=sample_space[attribute].unique()
        entropy_value=0
        # for each value calculate the gain and add all the gain values
        for value in unique_values:
            count=(sample_space[attribute] == value).sum()
            p=float(count) / rows_count
            attr_sample_space=sample_space.filter([attribute, 'output_value'], axis=1)
            attr_sample_space=attr_sample_space.loc[(attr_sample_space[attribute] == value)]
            attr_sample_space_rows_count=attr_sample_space['output_value'].count()
            true=(attr_sample_space['output_value'] == 1).sum()
            prob_true=float(true) / attr_sample_space_rows_count
            false=(attr_sample_space['output_value'] == 0).sum()
            prob_false=float(false) / attr_sample_space_rows_count
            if prob_true == 0:
                entropy_true=0
            else:
                entropy_true=prob_true * (np.log2(prob_true))
            if prob_false == 0:
                entropy_false=0
            else:
                entropy_false=prob_false * (np.log2(prob_false))
            total_entropy=-(entropy_false + entropy_true)
            entropy_value+=p * total_entropy
        if min_entropy == None or entropy_value < min_entropy:
            best_attribute=attribute
            min_entropy=entropy_value
            # return the best attribute
    return best_attribute


def build_decision_tree(sample_space, attributes, pvalue):
    global internal_nodes, leaves
    # if the sample has only positives values, return node with data = 'True'
    if (sample_space['output_value'] == 1).sum() == sample_space['output_value'].count():
        leaves+=1
        return TreeNode()
        # if the sample has only negative values, return node with data = 'False'
    if (sample_space['output_value'] == 0).sum() == sample_space['output_value'].count():
        leaves+=1
        return TreeNode('F')
    # if there are no attributes, build the node with positive or negative based on their count
    if len(attributes) == 0:
        true=0
        false=0
        true=(sample_space['output_value'] == 1).sum()
        false=(sample_space['output_value'] == 0).sum()
        if true >= false:
            leaves+=1
            return TreeNode()
        else:
            leaves+=1
            return TreeNode('F')
            # chose the best attribute based for which gain is max
    chosen_attribute=find_next_attribute(sample_space, attributes)
    # print('chosen_attribute: ', chosen_attribute)
    attributes.remove(chosen_attribute)
    node=None
    # calculate the p_value for the chosen attribute
    chisq=chisquare_criterion(sample_space, chosen_attribute)
    # build the node if the chi sqaure value if less than p_value, else terminate the node
    if chisq < pvalue:
        node=TreeNode(chosen_attribute + 1)
        internal_nodes+=1
        uniqueValues=sample_space[chosen_attribute].unique()
        i=1
        true_missing_val=-1
        false_missing_val=-1
        while i < 6:
            # build the child nodes for the chosen attribute node
            if i in uniqueValues:
                sample_space_subset=sample_space.loc[sample_space[chosen_attribute] == i]
                if sample_space_subset.empty:
                    true=(sample_space['output_value'] == 1).sum()
                    false=(sample_space['output_value'] == 0).sum()
                    if true >= false:
                        leaves+=1
                        node.nodes[i - 1]=TreeNode()
                    else:
                        leaves+=1
                        node.nodes[i - 1]=TreeNode('F')
                else:
                    attri=copy.deepcopy(attributes)
                    is_node=build_decision_tree(sample_space_subset, attributes, pvalue)
                    if is_node:
                        node.nodes[i - 1]=is_node
                    else:
                        true=(sample_space_subset['output_value'] == 1).sum()
                        false=(sample_space_subset['output_value'] == 0).sum()
                        if true >= false:
                            leaves+=1
                            node.nodes[i - 1]=TreeNode()
                        else:
                            leaves+=1
                            node.nodes[i - 1]=TreeNode('F')
            else:
                if true_missing_val == -1 and false_missing_val == -1:
                    true_missing_val=(sample_space['output_value'] == 1).sum()
                    false_missing_val=(sample_space['output_value'] == 0).sum()
                if true_missing_val >= false_missing_val:
                    leaves+=1
                    node.nodes[i - 1]=TreeNode()
                else:
                    leaves+=1
                    node.nodes[i - 1]=TreeNode('F')
            i+=1
    else:
        return None
    return node


# traverse the tree and return the best possible value for the test data
def classify(root, datapoint):
    if root.data == 'T': return 1
    if root.data == 'F': return 0
    return classify(root.nodes[datapoint[int(root.data) - 1] - 1], datapoint)


# parse the command line arguments
parser=argparse.ArgumentParser()
parser.add_argument('-p', help='specify p-value threshold', dest='pvalue', action='store', default='0.005')
parser.add_argument('-f1', help='specify training dataset path', dest='train_dataset', action='store', default='')
parser.add_argument('-f2', help='specify test dataset path', dest='test_dataset', action='store', default='')
parser.add_argument('-o', help='specify output file', dest='output_file', action='store', default='')
parser.add_argument('-t', help='specify decision tree', dest='decision_tree', action='store', default='')

args=vars(parser.parse_args())

# read the train data
train_data_file_name=args['train_dataset']
train_data_label_file=args['train_dataset'].split('.')[0] + '_label.csv'
sample_space=pd.read_csv(train_data_file_name, header=None, sep=" ")
train_output_values=pd.read_csv(train_data_label_file, header=None)
sample_space['output_value']=train_output_values[0]
attribute_count=sample_space.shape[1] - 1
attributes=[i for i in range(attribute_count)]
pvalue=float(args['pvalue'])

# build the tree using the train data
root=build_decision_tree(sample_space, attributes, pvalue)
root.save_tree(args['decision_tree'])

# read test data
test_data_file=args['test_dataset']
test_data_space=pd.read_csv(test_data_file, header=None, sep=" ")
test_row_count=test_data_space.shape[0]
test_result=[]

# get the best possible output value for the test data
for i in range(test_row_count):
    test_result.append([classify(root, test_data_space.loc[i])])

# write the test output to a file
output_file=args['output_file']
with open(output_file, "wb") as f:
    writer=csv.writer(f)
    writer.writerows(test_result)
print("internal nodes: ", internal_nodes, " leaf nodes: ", leaves)


