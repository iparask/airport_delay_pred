from __future__ import division
import warnings
warnings.filterwarnings('ignore')

import sys
import random
import numpy as np
from datetime import datetime
from sklearn import linear_model, cross_validation, metrics, svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import os

def memory_use():
    memfile = open('/proc/meminfo','r')
    total_mem_str = memfile.readline().split()
    used_mem_str = memfile.readline().split()
    memfile.close()

    total_mem = int(total_mem_str[1])
    used_mem = int(used_mem_str[1])
    return {'total':total_mem,'used':used_mem}


def accuracy_measure(predicted,known):

    correct_answers = 0
    for i in range(0,predicted.shape[0]):
        if (abs(predicted[i]-known[i])/known[i])<0.2:
            correct_answers = correct_answers + 1

    return {"relative":(correct_answers/known.shape[0]),"abs":correct_answers}



def read_csv_from_dir(path, cols, col_types=None):
    pieces = []
    for f in os.listdir(path):
        if f[0] != '.':
            fhandle = open(os.path.join(path,f),'r')
            pieces.append(pd.read_csv(fhandle, names=cols, dtype=col_types))
            fhandle.close()
    return pd.concat(pieces, ignore_index=True)


def forest_generation(number_of_trees,features,delay,nprocs=1):

    results = dict()
    start = datetime.now()
    clf_rf = RandomForestClassifier(n_estimators=number_of_trees, n_jobs=nprocs,min_samples_split=100)
    clf_rf.fit(features, delay)
    total_time = (datetime.now()-start).total_seconds()
    memory = sys.getsizeof(clf_rf)

    results = {"forest":clf_rf,"time":total_time,"memory":memory}
    return results

def predict(forest,samples):

    start = datetime.now()
    pr = forest.predict(samples)
    total_time = (datetime.now()-start).total_seconds()
    results = {"predictions":pr,"time":total_time}
    return results


if __name__ == '__main__':

    # read files
    logfile = open("training_with_past_conc.log",'w')
    #Number of runs
    for j in range(1,4):
        logfile.write("Run Number %d\n"%j)
        #Years that will be used
        train_y=[]
        train_x=[]
        years=[]
        #The years it is going to use as training data. Starts with 2013 and adds each previous year until it
        #reaches 2000.
        for i in range(2013,1999,-1):
            years.append(i)
            logfile.write ("Training with %s\n"%years)
            cols = ['delay', 'month', 'day', 'dow', 'hour', 'distance', 'carrier', 'dest', 'days_from_holiday']
            col_types = {'delay': int, 'month': int, 'day': int, 'dow': int, 'hour': int, 'distance': int, 
               'carrier': str, 'dest': str, 'days_from_holiday': int}
            data_2014 = read_csv_from_dir('../data/ord_2014_1', cols, col_types)
            data = read_csv_from_dir('../data/ord_%d_1'%i, cols, col_types)

            # Create training set and test set
            cols = ['month', 'day', 'dow', 'hour', 'distance', 'days_from_holiday']
            train_y.append(data['delay'])
            train_x.append(data[cols])
            test_y = data_2014['delay']
            test_x = data_2014[cols]

            pdtrain_x = pd.concat(train_x)
            pdtrain_y = pd.concat(train_y)

            membefore = memory_use()
            forest = forest_generation(number_of_trees=10,features=pdtrain_x,delay=pdtrain_y,nprocs=4)
            memafter = memory_use()

            logfile.write ("Training Time: %f Memory Used by forest: %d\n"%(forest["time"],(membefore['used']-memafter['used'])))

            # Evaluate on test set
            pred = predict(forest=forest["forest"],samples=test_x)
            predicted = pred['predictions']

            for i in range(0,predicted.shape[0]):
                if type(predicted[i]) != np.int64:
                    predicted[i]=0


            # print results
            #cm = confusion_matrix(test_y, pred["predictions"])
            #print("Confusion matrix")
            #print(pd.DataFrame(cm))
            #report_svm = precision_recall_fscore_support(list(test_y), list(pred["predictions"]), average='micro')
            #print "\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \(report_svm[0], report_svm[1], report_svm[2], accuracy_score(list(test_y), list(predicted)))
            accur = accuracy_measure(test_y,predicted)
            logfile.write("Accuracy: %f Absolute Number: %d\n\n\n"%(accur['relative'],accur['abs']))
            del forest
            del pred

    logfile.close()
