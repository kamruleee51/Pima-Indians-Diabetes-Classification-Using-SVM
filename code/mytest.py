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
from sklearn.metrics import roc_curve, auc
import pandas as pd
from tflearn.data_utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#%% Load The data to test the Model
'Here you can import your data and pass it to the loaded model'
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

#%% Select the new Test data to test the saved Model
Train, Test, TrainLabel, TestLabel = train_test_split(Data, NewLabel, test_size=0.3, random_state=100)

#%% Load the Model
'.............Model is loaded here..............................'
filename = 'myModel.pkl'
loaded_model = joblib.load(filename)
Accuracy = loaded_model.score(Test, TestLabel)
prob=loaded_model.predict_proba(Test)
print('Accuracy {} %'.format(100*Accuracy))
LabelTest_1Hot=to_categorical(TestLabel,2)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(LabelTest_1Hot[:, i], prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
AUCC=(roc_auc[0]+roc_auc[1])/2
print()
print("Area Under ROC (AUC) for nu-SVM: {}".format(AUCC))
print()
endTime = time.time()
print('It took {0:0.1f} seconds'.format(endTime - startTime))
#%%...................The END..........................