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
from sklearn.model_selection import GridSearchCV

#%% Read CSV Data and Pre-processing (standardized and split) of Data
RawData= pd.read_csv('hw3data.csv',header=None)
RawdataArray=np.array(RawData)

#==========Standardization of RawData========
standardized_RawdataArray = preprocessing.scale(RawdataArray[:,0:10])

#========Positive and Negative Splitt =========
PositiveData=standardized_RawdataArray[0:4000,:]
NegativeData=standardized_RawdataArray[4000:8000,:]

PositiveDataCase_1=PositiveData[0:800,:]
PositiveDataCase_2=PositiveData[800:1600,:]
PositiveDataCase_3=PositiveData[1600:2400,:]
PositiveDataCase_4=PositiveData[2400:3200,:]
PositiveDataCase_5=PositiveData[3200:4000,:]

# for the case-1
TestPosCase_1=PositiveDataCase_1
TrainValCase_1=np.concatenate((PositiveDataCase_2, PositiveDataCase_3,PositiveDataCase_4,PositiveDataCase_5), axis=0)
# for the case-2
TestPosCase_2=PositiveDataCase_2
TrainValCase_2=np.concatenate((PositiveDataCase_1, PositiveDataCase_3,PositiveDataCase_4,PositiveDataCase_5), axis=0)
# for the case-3
TestPosCase_3=PositiveDataCase_3
TrainValCase_3=np.concatenate((PositiveDataCase_1, PositiveDataCase_2,PositiveDataCase_4,PositiveDataCase_5), axis=0)
# for the case-4
TestPosCase_4=PositiveDataCase_4
TrainValCase_4=np.concatenate((PositiveDataCase_1, PositiveDataCase_2,PositiveDataCase_3,PositiveDataCase_5), axis=0)
# for the case-5
TestPosCase_5=PositiveDataCase_5
TrainValCase_5=np.concatenate((PositiveDataCase_1, PositiveDataCase_2,PositiveDataCase_3,PositiveDataCase_4), axis=0)

NegativeDataCase_1=NegativeData[0:800,:]
NegativeDataCase_2=NegativeData[800:1600,:]
NegativeDataCase_3=NegativeData[1600:2400,:]
NegativeDataCase_4=NegativeData[2400:3200,:]
NegativeDataCase_5=NegativeData[3200:4000,:]

# for the case-1
TestNegCase_1=NegativeDataCase_1
TrainValNegCase_1=np.concatenate((NegativeDataCase_2, NegativeDataCase_3,NegativeDataCase_4,NegativeDataCase_5), axis=0)

# for the case-2
TestNegCase_2=NegativeDataCase_2
TrainValNegCase_2=np.concatenate((NegativeDataCase_1, NegativeDataCase_3,NegativeDataCase_4,NegativeDataCase_5), axis=0)
# for the case-3
TestNegCase_3=NegativeDataCase_3
TrainValNegCase_3=np.concatenate((NegativeDataCase_1, NegativeDataCase_2,NegativeDataCase_4,NegativeDataCase_5), axis=0)
# for the case-4
TestNegCase_4=NegativeDataCase_4
TrainValNegCase_4=np.concatenate((NegativeDataCase_1, NegativeDataCase_2,NegativeDataCase_3,NegativeDataCase_5), axis=0)
# for the case-5
TestNegCase_5=NegativeDataCase_5
TrainValNegCase_5=np.concatenate((NegativeDataCase_1, NegativeDataCase_2,NegativeDataCase_3,NegativeDataCase_4), axis=0)

#==Concatenate Negative 1-4000 and Positive 4001-8000==
Data=np.concatenate((NegativeData, PositiveData), axis=0)
#for case-1
TrainValCase_1=np.concatenate((TrainValNegCase_1, TrainValCase_1), axis=0)
TestCase_1=np.concatenate((TestNegCase_1, TestPosCase_1), axis=0)
#for case-2
TrainValCase_2=np.concatenate((TrainValNegCase_2, TrainValCase_2), axis=0)
TestCase_2=np.concatenate((TestNegCase_2, TestPosCase_2), axis=0)
#for case-3
TrainValCase_3=np.concatenate((TrainValNegCase_3, TrainValCase_3), axis=0)
TestCase_3=np.concatenate((TestNegCase_3, TestPosCase_3), axis=0)
#for case-4
TrainValCase_4=np.concatenate((TrainValNegCase_4, TrainValCase_4), axis=0)
TestCase_4=np.concatenate((TestNegCase_4, TestPosCase_4), axis=0)
#for case-5
TrainValCase_5=np.concatenate((TrainValNegCase_5, TrainValCase_5), axis=0)
TestCase_5=np.concatenate((TestNegCase_5, TestPosCase_5), axis=0)

#======== Extract the Labels=============
label=RawdataArray[:,10]

#========Make the Label -1 to 0============
CopyLabel = label.copy()
CopyLabel[CopyLabel < 0] = 0

#=====Positive and Negative Split of the Label=====
LabelPositive=CopyLabel[0:4000]
LabelNegative=CopyLabel[4000:8000]

PositiveLabelCase_1=LabelPositive[0:800]
PositiveLabelCase_2=LabelPositive[800:1600]
PositiveLabelCase_3=LabelPositive[1600:2400]
PositiveLabelCase_4=LabelPositive[2400:3200]
PositiveLabelCase_5=LabelPositive[3200:4000]
# for the case-1
LabelTestPosCase_1=PositiveLabelCase_1
LabelTrainValPosCase_1=np.concatenate((PositiveLabelCase_2, PositiveLabelCase_3, PositiveLabelCase_4, PositiveLabelCase_5), axis=0)
# for the case-2
LabelTestPosCase_2=PositiveLabelCase_2
LabelTrainValPosCase_2=np.concatenate((PositiveLabelCase_1, PositiveLabelCase_3, PositiveLabelCase_4, PositiveLabelCase_5), axis=0)
# for the case-3
LabelTestPosCase_3=PositiveLabelCase_3
LabelTrainValPosCase_3=np.concatenate((PositiveLabelCase_1, PositiveLabelCase_2, PositiveLabelCase_4, PositiveLabelCase_5), axis=0)
# for the case-4
LabelTestPosCase_4=PositiveLabelCase_4
LabelTrainValPosCase_4=np.concatenate((PositiveLabelCase_1, PositiveLabelCase_2, PositiveLabelCase_3, PositiveLabelCase_5), axis=0)
# for the case-5
LabelTestPosCase_5=PositiveLabelCase_5
LabelTrainValPosCase_5=np.concatenate((PositiveLabelCase_1, PositiveLabelCase_2, PositiveLabelCase_3, PositiveLabelCase_4), axis=0)

NegativeLabelCase_1=LabelNegative[0:800]
NegativeLabelCase_2=LabelNegative[800:1600]
NegativeLabelCase_3=LabelNegative[1600:2400]
NegativeLabelCase_4=LabelNegative[2400:3200]
NegativeLabelCase_5=LabelNegative[3200:4000]

# for the case-1
LabelTestNegCase_1=NegativeLabelCase_1
LabelTrainValNegCase_1=np.concatenate((NegativeLabelCase_2, NegativeLabelCase_3, NegativeLabelCase_4, NegativeLabelCase_5), axis=0)
# for the case-2
LabelTestNegCase_2=NegativeLabelCase_2
LabelTrainValNegCase_2=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_3, NegativeLabelCase_4, NegativeLabelCase_5), axis=0)
# for the case-3
LabelTestNegCase_3=NegativeLabelCase_3
LabelTrainValNegCase_3=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_2, NegativeLabelCase_4, NegativeLabelCase_5), axis=0)
# for the case-3
LabelTestNegCase_3=NegativeLabelCase_3
LabelTrainValNegCase_3=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_2, NegativeLabelCase_4, NegativeLabelCase_5), axis=0)
# for the case-4
LabelTestNegCase_4=NegativeLabelCase_4
LabelTrainValNegCase_4=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_2, NegativeLabelCase_3, NegativeLabelCase_5), axis=0)
# for the case-5
LabelTestNegCase_5=NegativeLabelCase_5
LabelTrainValNegCase_5=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_2, NegativeLabelCase_3, NegativeLabelCase_4), axis=0)

#====Label concatenate. lebel 0 first then Label 1.====
NewLabel=np.concatenate((LabelNegative, LabelPositive), axis=0)
#for Case -1
TrainValLabelcase_1=np.concatenate((LabelTrainValNegCase_1, LabelTrainValPosCase_1), axis=0)
TestLabelcase_1=np.concatenate((LabelTestNegCase_1, LabelTestPosCase_1), axis=0)
#for Case -2
TrainValLabelcase_2=np.concatenate((LabelTrainValNegCase_2, LabelTrainValPosCase_2), axis=0)
TestLabelcase_2=np.concatenate((LabelTestNegCase_2, LabelTestPosCase_2), axis=0)
#for Case -3
TrainValLabelcase_3=np.concatenate((LabelTrainValNegCase_3, LabelTrainValPosCase_3), axis=0)
TestLabelcase_3=np.concatenate((LabelTestNegCase_3, LabelTestPosCase_3), axis=0)
#for Case -4
TrainValLabelcase_4=np.concatenate((LabelTrainValNegCase_4, LabelTrainValPosCase_4), axis=0)
TestLabelcase_4=np.concatenate((LabelTestNegCase_4, LabelTestPosCase_4), axis=0)
#for Case -5
TrainValLabelcase_5=np.concatenate((LabelTrainValNegCase_5, LabelTrainValPosCase_5), axis=0)
TestLabelcase_5=np.concatenate((LabelTestNegCase_5, LabelTestPosCase_5), axis=0)

#%% Train, Validation and Test set creation

Kernel='rbf' # here you can select the kernel.

tuned_parameters = [{'kernel': [Kernel],'nu':[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
#tuned_parameters = [{'kernel': [Kernel],'nu':[0.01,0.05]}]

classifier = svm.NuSVC()

TrainValidationList=[TrainValCase_1,TrainValCase_2, TrainValCase_3, TrainValCase_4, TrainValCase_5]

TrainValidationListLAbel=[TrainValLabelcase_1,TrainValLabelcase_2, TrainValLabelcase_3, TrainValLabelcase_4, TrainValLabelcase_5]

OptimumNU=[]
AUCvar=[]
for ind in range(len(TrainValidationList)):
    clf = GridSearchCV(classifier, tuned_parameters, cv=5,scoring='roc_auc',n_jobs=1)
    data=TrainValidationList[ind]
    dataLabel=TrainValidationListLAbel[ind]
    clf.fit(data, dataLabel)
    AUCvar.append(clf.cv_results_['mean_test_score'])
    OptimumNU.append(clf.best_params_)


Train=[]
TrainLabel=[]
Test=[TestCase_1,TestCase_2,TestCase_3,TestCase_4,TestCase_5]
TestLabel=[TestLabelcase_1,TestLabelcase_2,TestLabelcase_3,TestLabelcase_4,TestLabelcase_5]

for ind in range(len(TrainValidationList)):
    data=TrainValidationList[ind]
    dataLabel=TrainValidationListLAbel[ind]
    X_train, X_test, y_train, y_test = train_test_split(
    data, dataLabel, test_size=0.5, random_state=42)
    Train.append(X_train)
    TrainLabel.append(y_train)

maxAUC=[]
for ind in range(len(AUCvar)):
    maxAUC.append(np.amax(AUCvar[ind]))

maxx=maxAUC.index(max(maxAUC))
nucase=OptimumNU[maxx]
nucase_1=nucase['nu']

AROC=[]
classifier=svm.NuSVC(nu=nucase_1, kernel=Kernel,probability=True)
for ind in range(5):
    probas_ = classifier.fit(Train[ind], TrainLabel[ind]).predict_proba(Test[ind])
    LabelTest_1Hot= to_categorical(TestLabel[ind],2)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(LabelTest_1Hot[:, i], probas_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    AUCC=(roc_auc[0]+roc_auc[1])/2
    print()
    print("Area Under ROC (AUC) for SVM: {}".format(AUCC))
    AROC.append(AUCC)
    plt.figure()
    lw = 2
    plt.grid(True)
    plt.plot(fpr[1], tpr[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver operating characteristic (ROC) for nu-SVM')
    plt.show()

mean=np.mean(AROC)
std=np.std(AROC)
print("%0.3f (+/-%0.03f)" % (mean, std))

endTime = time.time()
print('It took {0:0.1f} seconds'.format(endTime - startTime))
#%%................THE END................