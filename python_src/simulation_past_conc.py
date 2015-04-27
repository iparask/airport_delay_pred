from __future__ import division
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import binascii
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import os

def memory_usage_ps():
    import subprocess
    out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],stdout=subprocess.PIPE).communicate()[0].split(b'\n')
    vsz_index = out[0].split().index(b'RSS')
    mem = float(out[1].split()[vsz_index])
    return mem


def accuracy_measure(predicted,known):

    correct_answers = 0
    for i in range(0,predicted.shape[0]):
        if (abs(predicted[i]-known[i]))<20:
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

    pr = np.zeros(samples.shape[0],np.int64)
    print "Predicting for ",samples.shape[0]
    start = datetime.now()
    for i in range(0,samples.shape[0]):
        pr[i] = forest.predict(samples.irow(i))
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
            print "Training with %s\n"%years
            cols = ['delay', 'month', 'day', 'dow', 'hour', 'distance', 'carrier', 'dest', 'days_from_holiday']
            col_types = {'delay': int, 'month': int, 'day': int, 'dow': int, 'hour': int, 'distance': int, 
               'carrier': str, 'dest': str, 'days_from_holiday': int}
            data_2014 = read_csv_from_dir('../data/ord_2014_1', cols, col_types)
            data = read_csv_from_dir('../data/ord_%d_1'%i, cols, col_types)

            # Create training set and test set
            cols = ['day','dow','distance','carrier','dest','days_from_holiday']

            carrier = data['carrier']
            dest = data['dest']

            carrier = pd.Series(carrier.ravel()).unique()
            dest = pd.Series(dest.ravel()).unique()

            for carr in carrier:
                numID=int(binascii.b2a_hex(carr),16)
                data['carrier'][data['carrier']==carr]=numID

            for dst in dest:
                numID=int(binascii.b2a_hex(dst),16)
                data['dest'][data['dest']==dst]=numID

            carrier_2014 = data_2014['carrier']
            dest_2014 = data_2014['dest']

            carrier_2014 = pd.Series(carrier_2014.ravel()).unique()
            dest_2014 = pd.Series(dest_2014.ravel()).unique()

            for carrier in carrier_2014:
                numID=int(binascii.b2a_hex(carrier),16)
                data_2014['carrier'][data_2014['carrier']==carrier]=numID

            for dest in dest_2014:
                numID=int(binascii.b2a_hex(dest),16)
                data_2014['dest'][data_2014['dest']==dest]=numID
            train_y.append(data['delay'])
            train_x.append(data[cols])
            test_y = data_2014['delay']
            test_x = data_2014[cols]

            pdtrain_x = pd.concat(train_x)
            pdtrain_y = pd.concat(train_y)
            print pdtrain_x.shape
            logfile.write ("Data Size %d %d\n"%(pdtrain_x.shape[0],pdtrain_x.shape[1]))

            membefore = memory_usage_ps()
            forest = forest_generation(number_of_trees=10,features=pdtrain_x,delay=pdtrain_y,nprocs=1)
            memafter = memory_usage_ps()

            logfile.write ("Training Time: %f Memory Used by forest: %d\n"%(forest["time"],(memafter-membefore)))
            print "Training Time: %f Memory Used by forest: %d\n"%(forest["time"],(memafter-membefore))

            # Evaluate on test set
            predicted = predict(forest=forest["forest"],samples=test_x)

            # print results
            accur = accuracy_measure(predicted["predictions"],test_y)
            logfile.write("Accuracy: %f Absolute Number: %d. Prediction Time %f\n\n\n"%(accur['relative'],accur['abs'],predicted["time"]))
            print "Accuracy: ",accur['relative'],"Absolute Number: ",accur['abs'],"Prediction Time ",predicted["time"]
            del forest
            del predicted
    logfile.close()
