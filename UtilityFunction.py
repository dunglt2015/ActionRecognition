# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:07:01 2017

@author: Destiny
"""
import numpy as np
import os

def printTreeStructure( treeInForest ):
    n_nodes = treeInForest.tree_.node_count
    children_left = treeInForest.tree_.children_left
    children_right = treeInForest.tree_.children_right
    feature = treeInForest.tree_.feature
    threshold = treeInForest.tree_.threshold
    
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
    
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    
    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print() 
    return None

def getForestLeafs( forest , number_tree ):
    max_leaf = 32
    leaf_list = np.zeros(shape= max_leaf * number_tree, dtype=int)
    for i in range(0, number_tree):
        n_nodes = forest.estimators_[i].tree_.node_count
        children_left = forest.estimators_[i].tree_.children_left
        children_right = forest.estimators_[i].tree_.children_right
    
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes)
        count = 0
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
        
            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                leaf_list[count + i * max_leaf] = node_id
                count += 1
        
        while count < max_leaf - 1:
            leaf_list[count + i * max_leaf] = 0
            count += 1
            
    return leaf_list
    
def convertRFOutputToHistogram(rf_output, leaf_list, max_depth, n_estimators):
    histogram = np.zeros(shape = n_estimators * 2 ** max_depth, dtype=float)
    for i in range(0, len(rf_output)):
        a_descriptor = rf_output[i]
        temp_histogram = np.zeros(shape = n_estimators * 2 ** max_depth, dtype=float)
        for j in range(0, n_estimators):
            for k in range(0, 2 ** max_depth):
                if (a_descriptor[j] == leaf_list[k + j * 2 ** max_depth]):
                    temp_histogram[k + j * 2 ** max_depth] = 1./ n_estimators   
                    break
        histogram = [x + y for x, y in zip(histogram, temp_histogram)]
    
    histogram[:] = [x / len(rf_output) for x in histogram]
    return histogram
    
def getAllTextFileFromFolder(dir):
    list_file = []
    for filename in os.listdir(dir):
        list_file.append(filename)
    return list_file

def removeTestActorFiles(list_file, actor_name, index_of_actor):
    new_list = []
    for i in range(0, len(list_file)):
        if actor_name[index_of_actor] not in list_file[i]:
            new_list.append(list_file[i])
    return new_list

def removeTrainActorFiles(list_file, actor_name, index_of_actor):
    new_list = []
    for i in range(0, len(list_file)):
        if actor_name[index_of_actor] in list_file[i]:
            new_list.append(list_file[i])
    return new_list
    