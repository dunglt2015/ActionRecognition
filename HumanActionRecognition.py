# -*- coding: utf-8 -*-
"""
Created on Sat May 13 13:51:12 2017

@author: Destiny - Lê Tuấn Dũng
"""
# New code for testing all actor ,follow leave-one-actor-out strategy
import numpy as np
import random as rand
import pickle as pickle
import operator as operator

# import some functions from scikit-learn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC

# create some functions which are useful to make the code clear. 

from UtilityFunction import getAllTextFileFromFolder
from UtilityFunction import removeTestActorFiles, removeTrainActorFiles
from UtilityFunction import convertRFOutputToHistogram,getForestLeafs

#Global variables
test_label_file_path = "E:\\A-DATA\\test-label\\labels.txt"

stip_samples_folder = "E:\\A-DATA\\all-stip-samples\\"
rf_train_folder = "E:\\A-DATA\\rf-train-data\\"
histogram_folder = "E:\\A-DATA\\histogram-train-svm\\"

number_descriptor = 1500
number_train_sample_per_action = 33
number_test_sample_per_action = 3

number_action = 12
number_actor = 12
number_camera = 4

#Random Forest parameters
n_estimators = 10 
max_features = "auto"
max_depth = 5
class_weight = "balanced"

#SVM parameters 
C = 1
probability = True

#define actor name
actor_name = ["alba", "amel", "andreas", "chiara", "clare", "daniel", "florian",
              "hedlena", "julien", "nicolas", "pao", "srikumar"]

###############################################################################
# Random 800 descriptors positive and negative for training Random Forest     #
###############################################################################
def randomDescriptorForTrainingRandomForest(index_of_actor, stip_samples_cam, 
                                            rf_train_cam):
    for i in range(1,number_action + 1 ):
        # positive descriptors
        pos_count = 0
        positive_folder = stip_samples_cam + "\\Action_" + str(i)
        temp_list_positive_file = getAllTextFileFromFolder(positive_folder)
        list_positive_file = removeTestActorFiles(temp_list_positive_file, 
                                                  actor_name, 
                                                  index_of_actor)
        rf_train_file_path = rf_train_cam + "\\train_rf_" + str(i) + ".txt"
        
        save_file = open(rf_train_file_path, "a")
        save_file.truncate()
        
        rf_label_file_path = rf_train_folder + "labels_" + str(i) + ".txt"
        file_label = open(rf_label_file_path, "a")
        file_label.truncate()
        
        for j in range(0,len(list_positive_file)):
            stip_file_path = positive_folder + '\\' + list_positive_file[j]
            read_file = open(stip_file_path, "r")
            read_file.readline()
            read_file.readline()
            read_file.readline()
            while True:
                if pos_count < number_descriptor: 
                    line = read_file.readline()
                    if line == "" :
                        read_file.close()
                        break
                    else:
                        save_file.write(line)
                        pos_count += 1
                        file_label.write("1\n")
                else: 
                    break
        # get negative descriptors by random 11 others actions
        list_action_random = []
        k = 1
        while len(list_action_random) < number_action - 1:
            if k != i:
                list_action_random.append(k)
            k += 1
        
        for j in range(0,len(list_action_random)):
            neg_count = 0
            negative_folder = stip_samples_cam + "\\Action_" + str(list_action_random[j])
            temp_list_positive_file = getAllTextFileFromFolder(negative_folder)
            list_negative_file = removeTestActorFiles(temp_list_positive_file, 
                                                  actor_name, 
                                                  index_of_actor)
            print(len(list_negative_file))
            while neg_count < 5 * (pos_count/(number_action - 1)):
                k = rand.choice(range(0,number_train_sample_per_action))
                stip_file_path = negative_folder + '\\' + list_negative_file[k]
                read_file = open(stip_file_path, "r")
                read_file.readline()
                read_file.readline()
                read_file.readline()
                while True:
                    if neg_count < 5 * (pos_count/(number_action - 1)): 
                        line = read_file.readline()
                        if line == "" :
                            read_file.close()
                            break
                        else:
                            save_file.write(line)
                            neg_count += 1
                            file_label.write("2\n")
                    else: 
                        break
        file_label.close()
        save_file.close()
    return None

###############################################################################
# Train Random Forest for each action                                         #
###############################################################################  
def trainRandomForestModel(rf_model_for_all_cam, rf_train_cam):
    rf_model_list = []
    
    for i in range(1,number_action + 1):     
        # RF data to train Random Forest
        data = open(rf_train_cam + "\\train_rf_" + str(i) + ".txt")
        file_label = open(rf_train_folder +"\\labels_" + str(i) + ".txt")
        examples = np.loadtxt(data)
        labels = np.loadtxt(file_label)
        data.close()
        file_label.close()
        X = examples[:, 7:]
        rf = RandomForestClassifier(n_estimators = n_estimators, 
                                    max_depth = max_depth,
                                    max_features = max_features,
                                    class_weight = class_weight)
        rf.fit(X,labels, None)
        rf_model_list.append(rf)
        
    rf_model_for_all_cam.append(rf_model_list)
    return rf_model_list

###############################################################################
# Using Random Forest train model to cluster and get Histogram                #
###############################################################################
def getHistogramByUsingRandomForestModel(rf_model_list, 
                                         stip_samples_cam,
                                         index_of_actor,
                                         histogram_cam_folder):
    for i in range(1,number_action +1):
        rf = rf_model_list[i-1]
        leaf_list = getForestLeafs(rf_model_list[i-1], n_estimators)
        data_file_path = histogram_cam_folder + "\\histogram_action_" + str(i) + ".txt"
        label_file_path = histogram_cam_folder +"\\labels_" + str(i) + ".txt"
        label_save_file = open(label_file_path, "a")
        label_save_file.truncate()
        save_file = open(data_file_path, "a")
        save_file.truncate()
        for j in range(1,number_action + 1):
            train_folder = stip_samples_cam + "\\Action_" +str(j)
            temp_list_train_sample = getAllTextFileFromFolder(train_folder)
            list_train_sample = removeTestActorFiles(temp_list_train_sample, 
                                                  actor_name, 
                                                  index_of_actor)
            
            for k in range(0,len(list_train_sample)):
                stip_file_path = train_folder + '\\' + list_train_sample[k]
                read_file = open(stip_file_path, "r")
                read_file.readline()
                read_file.readline()
                read_file.readline()
                file = read_file.readlines()
                features = np.loadtxt(file, ndmin=2)
                if features.size > 0:
                    X = features[:, 7:]
                    rf_output = rf.apply(X)
                    #print(str(rf_output))
                    histogram = convertRFOutputToHistogram(rf_output, 
                                                           leaf_list,
                                                           max_depth,
                                                           n_estimators)
                    #print(str(histogram))
                    for item in histogram:
                        save_file.write("%s\t" % item)
                    save_file.write("\n")
                    
                    if j == i:
                        label_save_file.write("1\n")
                    else:
                        label_save_file.write("0\n")
    
        save_file.close()
        label_save_file.close()
    return None

###############################################################################
# Training SVM                                                                #
###############################################################################

def trainSVM(histogram_cam_folder, svm_model_for_all_cam):
    svm_list = []
    for i in range(1, number_action + 1):
        data_file_path = histogram_cam_folder + "\\histogram_action_" + str(i) + ".txt"
        label_file_path = histogram_cam_folder +"\\labels_" + str(i) + ".txt"
        label_file = open(label_file_path, "r")
        training_sample_file = open(data_file_path, "r")
        X = np.loadtxt(training_sample_file)
        y = np.loadtxt(label_file)
        label_file.close()
        training_sample_file.close()
        
        clf = SVC(C = C, kernel = chi2_kernel, probability = probability, 
                  class_weight = class_weight)
        
        clf.fit(X,y)    
        s = pickle.dumps(clf)
        svm_list.append(s)
    svm_model_for_all_cam.append(svm_list)
    return None

def getListTestFile(test_stip_folder, test_label_save_file, index_of_actor):
    test_stip_file_list = []
    for i in range(1,number_action +1):
        folder_path = test_stip_folder + "\\Action_" + str(i)
        temp_list_test_sample = getAllTextFileFromFolder(folder_path)
        list_test_sample = removeTrainActorFiles(temp_list_test_sample, actor_name, index_of_actor)
        for j in range(0, len(list_test_sample)):
            temp_file = test_stip_folder + str(i) + "\\" +list_test_sample[j]
            test_label_save_file.write(str(i)+"\n")
            test_stip_file_list.append(temp_file)
        test_label_save_file.close()
    return test_stip_file_list

def trainingPhase(index_of_actor):
    #do some stuff here
    rf_model_for_all_cam = []
    svm_model_for_all_cam = []
    
    for t in range(0, number_camera):
        stip_samples_cam = stip_samples_folder + "Cam"+ str(t)
        rf_train_cam = rf_train_folder + "Cam"+ str(t)
        
        histogram_cam_folder = histogram_folder + "Cam" + str(t)
                                                                    
        randomDescriptorForTrainingRandomForest(index_of_actor, 
                                                stip_samples_cam, 
                                                rf_train_cam)
        rf_model_list = trainRandomForestModel(rf_model_for_all_cam, 
                               rf_train_cam)
        getHistogramByUsingRandomForestModel(rf_model_list, 
                                             stip_samples_cam,
                                             index_of_actor,
                                             histogram_cam_folder)         
        trainSVM(histogram_cam_folder, svm_model_for_all_cam)
    return None

def testingPhase(index_of_actor):
    #do some stuff here
    proba_for_all_cam = []
    for t in range(0, number_camera):
        test_stip_folder = stip_samples_folder + "\\Cam" + str(t)
        test_label_save_file = open(test_label_file_path, "a")
        test_label_save_file.truncate()
        test_stip_file_list =  getListTestFile(test_stip_folder, 
                                               test_label_save_file,
                                               index_of_actor)

        proba_per_cam = []
        for i in range(0, len(test_stip_file_list)):
            test_stip = open(test_stip_file_list[i], "r")
            test_stip.readline()
            test_stip.readline()
            test_stip.readline()
            file = test_stip.readlines()
            proba_all_act = []
            if(len(file) != 0):
                features = np.loadtxt(file, ndmin=2)
                X = features[:, 7:]
                #print(test_stip_file_list[i])
                for j in range(0, len(rf_model_for_all_cam[t])):
                    rf = rf_model_for_all_cam[t][j]
                    leaf_list = getForestLeafs(rf, n_estimators)
                    #print(i)
                    #print(j)
                    rf_output = rf.apply(X)
                    histogram = convertRFOutputToHistogram(rf_output, 
                                                           leaf_list,
                                                           max_depth,
                                                           n_estimators)
                    clf = pickle.loads(svm_model_for_all_cam[t][j])
                    predict_proba = clf.predict_proba(histogram)
                    #print(predict_proba)
                    proba_per_act = predict_proba[0][1]
                    proba_all_act.append(proba_per_act)
                proba_per_cam.append(proba_all_act)
                test_stip.close()
            else:
                proba_all_act = number_action * [0]
                proba_per_cam.append(proba_all_act)
        proba_for_all_cam.append(proba_per_cam)
    return None
# Main program
 
##############################################################################
# loop for testing all actors by leave-one-actor-out-strategy                #
##############################################################################

for m in range(0, number_actor):
    trainingPhase(m)
    result = testingPhase(m)
    print(result)
    

    
    
    
    
    