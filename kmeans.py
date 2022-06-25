#Implementing K-means 

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import random
start_time = datetime.now()

#Read in data as a CSV
df = pd.read_csv('/Users/chand/Desktop/School/Spring 2022/ML/Assignments/Programming3/545_cluster_dataset programming 3.txt'
                 , sep=r'\s{2,}',engine='python', header=None, names=['x','y'])

#Number of clusters
k = 5
#seed cluster starts
centroids = []
for i in range(0,k):
    num = random.randint(0,len(df))
    x = [ df['x'][num] , df['y'][num] ]
    centroids.append(x)
centroids = np.array(centroids)
#Vec to represent the classifications
classes = np.zeros(len(df))

#This function makes plots
def plotter(n):
    df['classes'] = classes.tolist()
    fig, ax = plt.subplots()
    colors = {0.0: 'green', 1.0: 'red', 2.0: 'blue', 3.0: 'black', 4.0: 'purple', 5.0: 'violet',6.0: 'orange',7.0: 'pink'}
    ax.scatter(df['x'], df['y'], s=3, c=df['classes'].map(colors))
    plt.title('Clusters= '+str(k)+', Iteration = '+str(n))

#This function performs the E step of EM, classification
def classify():
    for i in range(0,len(df)):
        norms = []
        for j in range(0,k):
            x = np.array([ df['x'][i], df['y'][i] ])
            xx = np.linalg.norm(x - centroids[j])
            norms.append(xx)
        min_val = min(norms)
        min_index = norms.index(min_val)
        classes[i]=min_index

#This function adjusts centroid, the M step of EM
def maximize():
    refs = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0])
    counts = []
    sums = []
    for i in range(0,k):
        ct = 0
        summ = np.array([0,0])
        for j in range(0,len(classes)):
            if classes[j] == refs[i]:
                ct+=1
                summ = np.array([ df['x'][j] , df['y'][j] ]) + summ
        counts.append(np.array([ct,ct]))
        sums.append(summ)
    counts = np.array(counts)
    cents = sums / counts
    return cents

#Add sum squares error!!!!

def sumsquare():
    refs = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0])
    avgs = []
    for i in range(0,k):
        cts = 0
        summ = np.array([0,0])
        for j in range(0,len(df)):
            if classes[j] == refs[i]:
                cts+=1
                summ = np.array([ df['x'][j] , df['y'][j] ]) + summ
        avgs.append(summ / cts)
    sumsq = 0
    for i in range(0,k):
        for j in range(0,len(df)):
            if classes[j] == refs[i]:
                sumsq += np.linalg.norm( np.array([ df['x'][j] , df['y'][j] ]) - avgs[i] )**2
    return sumsq


classify()
runs = 10
plotter(0)
for i in range(0,runs):
    classify()
    x = sumsquare()
    print(x)
    centroids = maximize()
    if i == (runs-1):
        plotter(i+1)

plt.show()
#Adjoin the classes vector and then plot




end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
