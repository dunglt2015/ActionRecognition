import numpy as np
import random as rand
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC
from UtilityFunction import getAllTextFileFromFolder,getForestLeafs,convertRFOutputToHistogram
import pickle
import operator


#Global variables
test_label_file_path = "E:\\A-DATA\\test-label\\labels.txt"
rf_label_file_path = "E:\\A-DATA\\rf-train-data\\labels_"
number_descriptor = 1500
number_train_sample_per_action = 33
number_test_sample_per_action = 3
number_action = 12
number_camera = 4

rf_model_for_all_cam = []
svm_model_for_all_cam = []

for t in range(0, number_camera):
    stip_folder = "E:\\A-DATA\\Test-Actor-4\\Cam"+ str(t) + "\\Action_"
    rf_train_data = "E:\\A-DATA\\rf-train-data\\Cam"+ str(t) + "\\train_rf_"
    histogram_file_path = "E:\\A-DATA\\histogram-train-svm\\Cam"+ str(t) + "\\histogram_action_"
    label_file_path = "E:\\A-DATA\\histogram-train-svm\\Cam"+ str(t) + "\\labels_"
    
    
    ###########################################################################
    # Random 800 descriptors positive and negative for training Random Forest #
    ###########################################################################
    
    
    for i in range(1,number_action + 1 ):
        # positive descriptors
        print(i)
        pos_count = 0
        list_positive_file = getAllTextFileFromFolder(stip_folder + str(i))
        print(stip_folder + str(i))
        data_file_path = rf_train_data + str(i) + ".txt"
        save_file = open(data_file_path, "a")
        save_file.truncate()
        file_label = open(rf_label_file_path + str(i) + ".txt", "a")
        file_label.truncate()
        
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
                        file_label.write("1\n")
                else: 
                    break
                
        # negative descriptors
        # random 11 others actions
        list_action_random = []
        k = 1
        while len(list_action_random) < number_action - 1:
            if k != i:
                list_action_random.append(k)
            k +=1
            
        
        for j in range(0,len(list_action_random)):
            neg_count = 0
            #print(j)
            list_negative_file = getAllTextFileFromFolder(stip_folder + str(list_action_random[j]))
            while neg_count < 5 * (pos_count/(number_action - 1)):
                k = rand.choice(range(0,number_train_sample_per_action))
                stip_file_path = stip_folder + str(list_action_random[j]) + '\\' + list_negative_file[k]
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
                 
    ###########################################################################
    # Train Random Forest for each action                                     #
    ###########################################################################   
    n_estimators = 10 
    max_features = "auto"
    max_depth = 5
    class_weight = "balanced"
    oob_score = True
    rf_model_list = []
    
    for i in range(1,number_action + 1):     
        # RF data to train Random Forest
        f = open(rf_train_data + str(i) + ".txt")
        file_label = open(rf_label_file_path + str(i) + ".txt")
        examples = np.loadtxt(f)
        labels = np.loadtxt(file_label)
        f.close()
        file_label.close()
        X = examples[:, 7:]
        rf = RandomForestClassifier(n_estimators = n_estimators, 
                                    max_depth = max_depth,
                                    max_features = max_features,
                                    oob_score = oob_score,
                                    class_weight = class_weight)
        rf.fit(X,labels, None)
        print(rf.oob_score_)
        rf_model_list.append(rf)
    
    rf_model_for_all_cam.append(rf_model_list)    
    
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
    svm_model_for_all_cam.append(svm_list)
   
    
 ###########################################################################
 # Testing SVM                                                            #
 ########################################################################### 
    # Get list stip test file
proba_for_all_cam = []
for t in range(0, number_camera):
    test_stip_folder = "E:\\A-DATA\\Test-Actor-4\\Test\\Cam"+ str(t) + "\\Action_"
    test_stip_file_list = [] 
    test_label_save_file = open(test_label_file_path, "a")
    test_label_save_file.truncate()
    for i in range(1,number_action +1):
        folder_path = test_stip_folder + str(i)
        list_test_sample = getAllTextFileFromFolder(folder_path)
        for j in range(0, len(list_test_sample)):
            temp_file = test_stip_folder + str(i) + "\\" +list_test_sample[j]
            #check_empty_file = open(temp_file, "r")
            #if sum(1 for _ in check_empty_file) > 3:
            test_label_save_file.write(str(i)+"\n")
            #check_empty_file.close()
            test_stip_file_list.append(temp_file)
    test_label_save_file.close()
    
    # Test  samples
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
#print(proba_for_all_cam)
    
    
sum_proba_per_sample = []
for t in range(0, number_test_sample_per_action * number_action):
    sum_all_cam = [0] * number_action
    for i in range(0, number_camera): 
        sum_all_cam = tuple(map(operator.add, sum_all_cam, proba_for_all_cam[i][t]))
    sum_proba_per_sample.append(sum_all_cam)

predict = []
for i in range(0, len(sum_proba_per_sample)):
    index, value = max(enumerate(sum_proba_per_sample[i]), key=operator.itemgetter(1))
    predict.append(index + 1)
print(predict)