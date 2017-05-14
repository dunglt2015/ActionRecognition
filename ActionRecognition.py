import numpy as np
import random as rand
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC
from UtilityFunction import getAllTextFileFromFolder,createLabelRFTrain,getForestLeafs,convertRFOutputToHistogram
import pickle

np.set_printoptions(threshold='nan')
stip_folder = "E:\\A-DATA\\stip-extract-files\\Cam3\\Action_"
test_stip_folder = "E:\\A-DATA\\stip-extract-files\\Test\\Cam3\\Action_"
rf_train_data = "E:\\A-DATA\\rf-train-data\\Cam3\\train_rf_"
histogram_file_path = "E:\\A-DATA\\histogram-train-svm\\Cam3\\histogram_action_"
label_file_path = "E:\\A-DATA\\histogram-train-svm\\Cam3\\labels_"
test_label_file_path = "E:\\A-DATA\\test-label\\labels.txt"
number_descriptor = 600
number_neg_class = 8
number_train_sample_per_action = 33
number_test_sample_per_action = 3
number_action = 12

###########################################################################
# Random 800 descriptors positive and negative for training Random Forest #
###########################################################################


for i in range(1,number_action + 1 ):
    # positive descriptors
    pos_count = 0
    list_positive_file = getAllTextFileFromFolder(stip_folder + str(i))
    data_file_path = rf_train_data + str(i) + ".txt"
    save_file = open(data_file_path, "a")
    save_file.truncate()
    
    for j in range(0,len(list_positive_file)):
        stip_file_path = stip_folder + str(i) + '\\' + list_positive_file[j]
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
            else: 
                break
            
    # negative descriptors
    # random number_neg_class = 8 others actions
    list_action_random = []
    while len(list_action_random) < number_neg_class:
        k = rand.choice(range(1,number_action + 1))
        if k != i:
            list_action_random.append(k)  
    
    for j in range(0,len(list_action_random)):
        neg_count = 0
        list_negative_file = getAllTextFileFromFolder(stip_folder + str(list_action_random[j]))
        while neg_count < (number_descriptor/number_neg_class):
            k = rand.choice(range(0,number_train_sample_per_action))
            stip_file_path = stip_folder + str(list_action_random[j]) + '\\' + list_negative_file[k]
            read_file = open(stip_file_path, "r")
            read_file.readline()
            read_file.readline()
            read_file.readline()
            while True:
                if neg_count < (number_descriptor/number_neg_class): 
                    line = read_file.readline()
                    if line == "" :
                        read_file.close()
                        break
                    else:
                        save_file.write(line)
                        neg_count += 1
                else: 
                    break
    
    save_file.close()
               
###########################################################################
# Train Random Forest for each action                                     #
###########################################################################   
n_estimators = 10 
max_features = "auto"
max_depth = 5
labels = createLabelRFTrain(number_descriptor)
rf_model_list = []

for i in range(1,number_action + 1):     
    # RF data to train Random Forest
    f = open(rf_train_data + str(i) + ".txt")
    examples = np.loadtxt(f)
    X = examples[:, 7:]
    rf = RandomForestClassifier(n_estimators = n_estimators, 
                                max_depth = max_depth,
                                max_features = max_features)
    rf.fit(X,labels, None)
    rf_model_list.append(rf)
    

###########################################################################
# Using Random Forest train model to cluster and get Histogram            #                         #
########################################################################### 
for i in range(1,number_action +1):
    rf = rf_model_list[i-1]
    leaf_list = getForestLeafs(rf_model_list[i-1], n_estimators)
    data_file_path = histogram_file_path + str(i) + ".txt"
    label_file_path_to_save = label_file_path + str(i) + ".txt"
    label_save_file = open(label_file_path_to_save, "a")
    label_save_file.truncate()
    save_file = open(data_file_path, "a")
    save_file.truncate()
    for j in range(1,number_action + 1):
        list_train_sample = getAllTextFileFromFolder(stip_folder + str(j))
       
        
        for k in range(0,len(list_train_sample)):
            stip_file_path = stip_folder + str(j) + '\\' + list_train_sample[k]
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
        
###########################################################################
# Training SVM                                                            #
###########################################################################
svm_list = []
C = 1
#kernel = ""
class_weight= "balanced"


for i in range(1, number_action + 1):
    data_file_path = histogram_file_path + str(i) + ".txt"
    label_file_path_to_save = label_file_path + str(i) + ".txt"
    label_file = open(label_file_path_to_save, "r")
    training_sample_file = open(data_file_path, "r")
    X = np.loadtxt(training_sample_file)
    y = np.loadtxt(label_file)
    clf = SVC(C = C, kernel = chi2_kernel, probability = True, 
              class_weight = class_weight)
    clf.fit(X,y)    
    label_predict = clf.predict_proba(X)
    #print(clf.score(X,y))
    #print(str(label_predict))
    #print("Number of mislabeled points : %d" % (y != label_predict).sum())
    s = pickle.dumps(clf)
    svm_list.append(s)
    label_file.close()
    training_sample_file.close()
    
###########################################################################
# Testing SVM                                                            #
########################################################################### 
# Get list stip test file
test_stip_file_list = [] 
test_label_save_file = open(test_label_file_path, "a")
test_label_save_file.truncate()
for i in range(1,number_action +1):
    folder_path = test_stip_folder + str(i)
    list_test_sample = getAllTextFileFromFolder(test_stip_folder + str(i))
    for j in range(0, len(list_test_sample)):
        temp_file = test_stip_folder + str(i) + "\\" +list_test_sample[j]
        check_empty_file = open(temp_file, "r")
        if sum(1 for _ in check_empty_file) > 3:
            test_label_save_file.write(str(i)+"\n")
            check_empty_file.close()
            test_stip_file_list.append(temp_file)
test_label_save_file.close()

# Test  samples
predict = []
for i in range(0, len(test_stip_file_list)):
    test_stip = open(test_stip_file_list[i], "r")
    test_stip.readline()
    test_stip.readline()
    test_stip.readline()
    file = test_stip.readlines()
    features = np.loadtxt(file, ndmin=2)
    X = features[:, 7:]
    print(test_stip_file_list[i])
    temp_predict = 0
    max_proba = 0
    for j in range(0, len(rf_model_list)):
        rf = rf_model_list[j]
        leaf_list = getForestLeafs(rf_model_list[j], n_estimators)
        print(i)
        print(j)
        rf_output = rf.apply(X)
        histogram = convertRFOutputToHistogram(rf_output, 
                                               leaf_list,
                                               max_depth,
                                               n_estimators)
        clf = pickle.loads(svm_list[j])
        predict_proba = clf.predict_proba(histogram)
        print(predict_proba)
        if(predict_proba[0][1] >= max_proba):
            temp_predict = j+1
            max_proba = predict_proba[0][1]
            
    print(max_proba)
    predict.append(temp_predict)
    test_stip.close()

print(predict)
