# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:31:59 2018

@author: Md. Kamrul Hasan
"""
#%% Import all the Required Libraries
print(__doc__)
import time
startTime = time.time()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn import svm
from tflearn.data_utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#%% Read CSV Data and Pre-processing (standardized and split) of Data
RawData= pd.read_csv('hw3data.csv',header=None)
RawdataArray=np.array(RawData)

#==========Standardization of RawData========
standardized_RawdataArray = preprocessing.scale(RawdataArray[:,0:10])

#========Positive and Negative Splitt =========
PositiveData=standardized_RawdataArray[0:4000,:]
NegativeData=standardized_RawdataArray[4000:8000,:]

#==Concatenate Negative 1-4000 and Positive 4001-8000==
Data=np.concatenate((NegativeData, PositiveData), axis=0)

#======== Extract the Labels=============
label=RawdataArray[:,10]

#========Make the Label -1 to 0============
CopyLabel = label.copy()
CopyLabel[CopyLabel < 0] = 0

#=====Positive and Negative Split of the Label=====
LabelPositive=CopyLabel[0:4000]
LabelNegative=CopyLabel[4000:8000]

#====Label concatenate. lebel 0 first then Label 1.====
NewLabel=np.concatenate((LabelNegative, LabelPositive), axis=0)

#%%Creat New data all 4 cases
AUCArraypoly=[]
AccracyArraypoly=[]
AUCArrayrbf=[]
AccracyArrayrbf=[]
listPercent=[0.1, 0.12,0.2,0.25,0.33,0.5]
for percentage in range(len(listPercent)):
    TrainData80Percent, TestData20Percent, TrainLabel80Percent, TestLabel20Percent = train_test_split(Data, NewLabel, test_size=listPercent[percentage], random_state=100)

    #%% select Hyperparameters comes from the Problem 3.1.
    Kernelpoly='poly'
    Best_nupoly=0.5

    Train=[TrainData80Percent]
    Trainlabel=[TrainLabel80Percent]
    Test=[TestData20Percent]
    Testlabel=[TestLabel20Percent]

    classifier=svm.NuSVC(nu=Best_nupoly, kernel=Kernelpoly,probability=True)
    for ind in range(len(Train)):
        classifier = classifier.fit(Train[ind], Trainlabel[ind])
        Predicted_Prob=classifier.predict_proba(Test[ind])
        LabelTest_1Hot= to_categorical(Testlabel[ind],2)
        print('---------------For Poly----------------------------')
        Accuracy=np.mean(Testlabel[ind] == np.argmax(Predicted_Prob, axis=1))
        print("Accuracy of nu-SVM: {}%".format(100*Accuracy))
        AccracyArraypoly.append(Accuracy)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(LabelTest_1Hot[:, i], Predicted_Prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        AUCC=(roc_auc[0]+roc_auc[1])/2
        AUCArraypoly.append(AUCC)
        print()
        print("Area Under ROC (AUC) for nu-SVM: {}".format(AUCC))
        print('-------------------------------------------------------')
        #%%..............................
            #%% select Hyperparameters comes from the Problem 3.1.
    Kernelrbf='rbf'
    Best_nurbf=0.3

    Train=[TrainData80Percent]
    Trainlabel=[TrainLabel80Percent]
    Test=[TestData20Percent]
    Testlabel=[TestLabel20Percent]

    classifier=svm.NuSVC(nu=Best_nurbf, kernel=Kernelrbf,probability=True)
    for ind in range(len(Train)):
        classifier = classifier.fit(Train[ind], Trainlabel[ind])
        Predicted_Prob=classifier.predict_proba(Test[ind])
        LabelTest_1Hot= to_categorical(Testlabel[ind],2)
        print('------------------For RBF--------------------------')
        Accuracy=np.mean(Testlabel[ind] == np.argmax(Predicted_Prob, axis=1))
        print("Accuracy of nu-SVM: {}%".format(100*Accuracy))
        AccracyArrayrbf.append(Accuracy)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(LabelTest_1Hot[:, i], Predicted_Prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        AUCC=(roc_auc[0]+roc_auc[1])/2
        AUCArrayrbf.append(AUCC)
        print()
        print("Area Under ROC (AUC) for nu-SVM: {}".format(AUCC))
        print('-------------------------------------------------------')

plt.figure()
plt.plot(listPercent,AUCArraypoly,'r--',listPercent,AccracyArraypoly,'b--')
plt.ylabel('Accuracy and AUC for poly')
plt.xlabel('Percentage of Testing')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(listPercent,AUCArrayrbf,'r--',listPercent,AccracyArrayrbf,'b--')
plt.ylabel('Accuracy and AUC for rbf')
plt.xlabel('Percentage of Testing')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(listPercent,AUCArrayrbf,'r--',listPercent,AUCArraypoly,'b--')
plt.ylabel('AUC for rbf and poly')
plt.xlabel('Percentage of Testing')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(listPercent,AccracyArrayrbf,'r--',listPercent,AccracyArraypoly,'b--')
plt.ylabel('Accuracy for rbf and poly')
plt.xlabel('Percentage of Testing')
plt.grid(True)
plt.show()
#%% save the best model
BestKernel='rbf'
Best_nu=0.3
BestKFold=5
Train, Test, TrainLabel, TestLabel = train_test_split(Data, NewLabel, test_size=(1/BestKFold), random_state=100)
classifier=svm.NuSVC(nu=Best_nu, kernel=BestKernel,probability=True)
classifier = classifier.fit(Train, TrainLabel)
filename = 'myModel.pkl'
joblib.dump(classifier, filename)
#loaded_model = joblib.load(filename)
#result = loaded_model.score(Test, TestLabel)
#print(result)
endTime = time.time()
print('It took {0:0.1f} seconds'.format(endTime - startTime))
#%%................THE END................