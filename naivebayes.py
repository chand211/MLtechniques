#Implementing Naive Bayes to classify spam email

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
start_time = datetime.now()

#Read in data as a CSV
df = pd.read_csv('/Users/chand/Desktop/School/Spring 2022/ML/Assignments/Programming2/spambase/spambase1.csv')
#Make dataframe of just rows labeled 1 and then shuffle rows
df1s = df.drop(range(1813,4601))
df1s = df1s.reindex(np.random.permutation(df1s.index))
df1s = df1s.reset_index(drop=True)
#Make dataframe of just rows labeled 0 and then shuffle rows
df0s = df.drop(range(0,1813))
df0s = df0s.reindex(np.random.permutation(df0s.index))
df0s = df0s.reset_index(drop=True)



#Reconstruct sorted dataframes
#This makes the train df
df1s1 = df1s.drop(range(0,906))
df1s1 = df1s1.reset_index(drop=True)
df0s1 = df0s.drop(range(0,1394))
df0s1 = df0s1.reset_index(drop=True)
#This makes the test df
df1s2 = df1s.drop(range(906,1813))
df1s2 = df1s2.reset_index(drop=True)
df0s2 = df0s.drop(range(1394,2788))
df0s2 = df0s2.reset_index(drop=True)

df_train = pd.concat([df1s1, df0s1], axis=0)
df_train = df_train.reset_index(drop=True)
df_test = pd.concat([df1s2, df0s2], axis=0)
df_test = df_test.reset_index(drop=True)

#Function that makes array of [mean,std] for each column in the training data
def MeanStd():
    zeros = []
    ones = []
    for i in range(0,57):
        s = np.std(df0s1.iloc[:,i])
        if s < 0.0001:
            s = 0.0001
        x = [np.mean(df0s1.iloc[:,i]), s]
        zeros.append(x)
    for i in range(0,57):
        s = np.std(df1s1.iloc[:,i])
        if s < 0.0001:
            s = 0.0001
        x = [np.mean(df1s1.iloc[:,i]), s]
        ones.append(x)
    return np.array(zeros), np.array(ones)

#This function calculates the proportion of spam to not spam in a data set
def P1(df):
    ctr0 = 0
    ctr1 = 0
    for i in range(0,len(df)):
        if df.iloc[i][57] == 1:
            ctr1 +=1
        elif df.iloc[i][57] == 0:
            ctr0 +=1
    return ctr0/len(df), ctr1/len(df)

#This function calculates the Gaussian classifier function
def Norm(x,mu,sig):
    co = 1 / ( np.sqrt(2*np.pi)*sig )
    expo = ( (x-mu)**2 ) / (2*(sig**2))
    return co*np.exp(-expo)

#This function classifies each data point, making sure to use logs to avoid underflow
def classify(ind):
    class0 = np.log(x0train)
    for i in range(0,57):
        num = Norm(df_test.iloc[ind][i],zeros[i][0], zeros[i][1])
        if num == 0:
            num = 0.0001
        class0 += np.log(num)
    class1 = np.log(x1train)
    for i in range(0,57):
        num = Norm(df_test.iloc[ind][i],ones[i][0], ones[i][1])
        if num == 0:
            num = 0.0001
        class1 += np.log(num)
    x = 0
    if class1 > class0:
        x = 1
    true = df_test.iloc[ind][57]
    finact.append(true)
    finpred.append(x)
    return np.array([x,true])

#Executing functions to prepare to classify data
x0train,x1train = P1(df_train)
zeros, ones = MeanStd()

#This for loop runs through the test data and classifies
count = 0
finact = []
finpred = []
for i in range(0,len(df_test)):
    vec = classify(i)
    vec = vec.tolist()
    if i % 10 == 0:
        print(round((i/len(df_test))*100,1),"%")
    if vec[0] == vec[1]:
        count += 1


#Confusion Matrix
data = {'y_Actual':    finact,
        'y_Predicted': finpred
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

confusion_matrix.to_csv('/Users/chand/Desktop/confusionmatrix.csv')

print(count/len(df_test))


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
