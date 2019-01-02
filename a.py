# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:07:17 2017

@author: Ruben
"""

import numpy as num;
import scipy as sci;
import csv;
import matplotlib.pyplot as plt;
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils


def dataInit():
    #Reads the .csv files in order to initialize the data
    
    #For the train set
    fop = open(r'train.csv',encoding = 'utf-8')
    
    trainReader = csv.reader(fop, delimiter=',', quotechar='"') #To take out the header
    
    trainHeader = trainReader.__next__(); 
    
    print(trainHeader)
    
    eofFlag = False #Flag for end of file
    
    trainInfo = [];
    
    while (eofFlag == False):
        try:
            trainInfo.append(trainReader.__next__());
        except StopIteration:         #Will be thrown at the end of the file
            eofFlag = True;
            
    fop.close();   
        
    #Same procedure for the test set
    fop = open(r'test.csv',encoding = 'utf-8')
    
    testReader = csv.reader(fop, delimiter=',', quotechar='"')
    
    testHeader = testReader.__next__();
    
    print(testHeader)
    
    eofFlag = False
    
    testInfo = [];
    
    while (eofFlag == False):
        try:
            testInfo.append(testReader.__next__());
        except StopIteration:        
            eofFlag = True;
            
    fop.close();   
    
    return trainInfo, testInfo

def dfPreProcessingTrain(trainInfo):
    #Processes a list given by dataInit into a DataFrame, converting the
    #data into suitable types (namely numeric)
    
    dfTrainInfo = pd.DataFrame(trainInfo);
    
    #Assign column names for improved readbility                          
    dfTrainInfo.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    
    #Convert everything that can be converted to numeric.
    dfTrainInfo['Survived']=num.round(pd.to_numeric(dfTrainInfo['Survived'],errors='coerce'))
    dfTrainInfo['Pclass']=num.round(pd.to_numeric(dfTrainInfo['Pclass'],errors='coerce'))
    dfTrainInfo['Age']=num.round(pd.to_numeric(dfTrainInfo['Age'],errors='coerce'))
    dfTrainInfo['SibSp']=pd.to_numeric(dfTrainInfo['SibSp'],errors='coerce')
    dfTrainInfo['Parch']=pd.to_numeric(dfTrainInfo['Parch'],errors='coerce')
    return dfTrainInfo

def dfPreProcessingTest(testInfo):
    #Same as dfPreProcessingTrain but for the test set
    #The main difference is that the TestSet does not have survival data.
    
    dfTestInfo = pd.DataFrame(testInfo);
    dfTestInfo.columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    dfTestInfo['Pclass']=num.round(pd.to_numeric(dfTestInfo['Pclass'],errors='coerce'))
    dfTestInfo['Age']=num.round(pd.to_numeric(dfTestInfo['Age'],errors='coerce'))
    dfTestInfo['SibSp']=pd.to_numeric(dfTestInfo['SibSp'],errors='coerce')
    dfTestInfo['Parch']=pd.to_numeric(dfTestInfo['Parch'],errors='coerce')
    return dfTestInfo

def assignSex(dfTrainInfo):
    #Assign a number for each sex. 1 if the passenger is male and 2 if it's female
    
    #Divide the dataframe into two
    dfMales = dfTrainInfo.loc[dfTrainInfo['Sex']=='male']
    dfFemales = dfTrainInfo.loc[dfTrainInfo['Sex']=='female']
    
    #Substitute the value in the Sex column
    dfMales['Sex'] = 1
    dfFemales['Sex'] = 2
    
    #Put everything together into another DataFrame (Probably there are better ways to do this...)
    frames = [dfMales,dfFemales]
    dfSexDivided = pd.concat(frames)
    dfSexDivided = dfSexDivided.sort_index()
    
    return dfSexDivided

#Main program

#Read the .csv files
trainInfo, testInfo = dataInit();
                        
#Convert them to dataframes
dfTrainInfo = dfPreProcessingTrain(trainInfo);
dfTestInfo = dfPreProcessingTest(testInfo);  

#For the train set:

#Convert Sex to a number                                
dfTrainInfoSex = assignSex(dfTrainInfo);

#Select features for the ANN                                
#dfTrainFeatures = dfTrainInfoSex[['Pclass','Sex','Age','SibSp','Parch']]  
dfTrainFeatures = dfTrainInfoSex[['Pclass','Sex','Parch']]

#Select the label. In this case the labels are survival or not  
dfTrainLabels = dfTrainInfoSex['Survived']

#Convert to array for processing
arrayTrainFeatures = num.array(dfTrainFeatures)
arrayTrainLabels = num.array(dfTrainLabels)


#Same with the test set (no labels were given, only features)

dfTestInfoSex = assignSex(dfTestInfo);
                         
#dfTestFeatures = dfTestInfoSex[['Pclass','Sex','Age','SibSp','Parch']]  
dfTestFeatures = dfTestInfoSex[['Pclass','Sex','Parch']]  

arrayTestFeatures = num.array(dfTestFeatures)

#ANN Properties

nb_epoch = 2000 #Probably too high, but it works
batch_size = 32

#Assign an average value if an age is not known (they only have some NaNs in Ages)
arrayTrainFeatures[num.isnan(arrayTrainFeatures)] = 29
arrayTestFeatures[num.isnan(arrayTestFeatures)] = 29
                  
#Define the ANN
annModel = Sequential() #Sequential ANN

#Layers of the ANN (Dense in this case)
annModel.add(Dense(30, input_dim=3, activation='sigmoid'))
annModel.add(Dense(3, input_dim=30, activation='sigmoid'))
annModel.add(Dense(1, input_dim=3, activation='sigmoid'))
annModel.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Get the ANN parameters                   
annModel.fit(arrayTrainFeatures, arrayTrainLabels,
              validation_split=0.2,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True)

#Predict survival (taking 0.5 as the cutoff)
TestPred = num.round(annModel.predict(arrayTestFeatures))

#Copying everything to a DataFrame
dfSurvived = dfTestInfo;
dfSurvived['Survived'] = TestPred.astype(int);          

#Finally write the DataFrame to a .csv file          
dfSurvived.to_csv(path_or_buf='result.csv',index=False,columns=['PassengerId','Survived'])
