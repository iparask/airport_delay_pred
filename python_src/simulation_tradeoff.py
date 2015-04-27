from __future__ import division
import warnings
warnings.filterwarnings('ignore')

import sys
import itertools as it
import binascii
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
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
    clf_rf = RandomForestClassifier(n_estimators=number_of_trees, n_jobs=nprocs, min_samples_split=100)
    clf_rf.fit(features, delay)
    total_time = (datetime.now()-start).total_seconds()
    memory = sys.getsizeof(clf_rf)

    results = {"forest":clf_rf,"time":total_time,"memory":memory}
    return results

def predict(forest,samples):

    pr = np.zeros(samples.shape[0],np.int64)
    start = datetime.now()
    for i in range(0,samples.shape[0]):
        pr[i] = forest.predict(samples.irow(i))
    total_time = (datetime.now()-start).total_seconds()
    results = {"predictions":pr,"time":total_time}
    return results


if __name__ == '__main__':

    NPROCS = 1
    TREES_NUM=10

    # read files
    cols = ['delay', 'month', 'day', 'dow', 'hour', 'distance', 'carrier', 'dest', 'days_from_holiday']
    col_types = {'delay': int, 'month': int, 'day': int, 'dow': int, 'hour': int, 'distance': int, 'carrier': str, 'dest': str, 'days_from_holiday': int}
    data_2009 = read_csv_from_dir('../data/ord_2009_1', cols, col_types)
    data_2010 = read_csv_from_dir('../data/ord_2010_1', cols, col_types)
    data_2011 = read_csv_from_dir('../data/ord_2011_1', cols, col_types)
    data_2012 = read_csv_from_dir('../data/ord_2012_1', cols, col_types)
    data_2013 = read_csv_from_dir('../data/ord_2013_1', cols, col_types)
    data_2014 = read_csv_from_dir('../data/ord_2014_1', cols, col_types)

    cols = ['day','dow','distance','carrier','dest','days_from_holiday']

    carrier_2009 = data_2009['carrier']
    dest_2009 = data_2009['dest']

    carrier_2009 = pd.Series(carrier_2009.ravel()).unique()
    dest_2009 = pd.Series(dest_2009.ravel()).unique()

    for carrier in carrier_2009:
        numID=int(binascii.b2a_hex(carrier),16)
        data_2009['carrier'][data_2009['carrier']==carrier]=numID

    for dest in dest_2009:
        numID=int(binascii.b2a_hex(dest),16)
        data_2009['dest'][data_2009['dest']==dest]=numID

    carrier_2010 = data_2010['carrier']
    dest_2010 = data_2010['dest']

    carrier_2010 = pd.Series(carrier_2010.ravel()).unique()
    dest_2010 = pd.Series(dest_2010.ravel()).unique()

    for carrier in carrier_2010:
        numID=int(binascii.b2a_hex(carrier),16)
        data_2010['carrier'][data_2010['carrier']==carrier]=numID

    for dest in dest_2010:
        numID=int(binascii.b2a_hex(dest),16)
        data_2010['dest'][data_2010['dest']==dest]=numID

    carrier_2011 = data_2011['carrier']
    dest_2011 = data_2011['dest']

    carrier_2011 = pd.Series(carrier_2011.ravel()).unique()
    dest_2011 = pd.Series(dest_2011.ravel()).unique()

    for carrier in carrier_2011:
        numID=int(binascii.b2a_hex(carrier),16)
        data_2011['carrier'][data_2011['carrier']==carrier]=numID

    for dest in dest_2011:
        numID=int(binascii.b2a_hex(dest),16)
        data_2011['dest'][data_2011['dest']==dest]=numID

    carrier_2012 = data_2012['carrier']
    dest_2012 = data_2012['dest']

    carrier_2012 = pd.Series(carrier_2012.ravel()).unique()
    dest_2012 = pd.Series(dest_2012.ravel()).unique()

    for carrier in carrier_2012:
        numID=int(binascii.b2a_hex(carrier),16)
        data_2012['carrier'][data_2012['carrier']==carrier]=numID

    for dest in dest_2012:
        numID=int(binascii.b2a_hex(dest),16)
        data_2012['dest'][data_2012['dest']==dest]=numID

    carrier_2013 = data_2013['carrier']
    dest_2013 = data_2013['dest']

    carrier_2013 = pd.Series(carrier_2013.ravel()).unique()
    dest_2013 = pd.Series(dest_2013.ravel()).unique()

    for carrier in carrier_2013:
        numID=int(binascii.b2a_hex(carrier),16)
        data_2013['carrier'][data_2013['carrier']==carrier]=numID

    for dest in dest_2013:
        numID=int(binascii.b2a_hex(dest),16)
        data_2013['dest'][data_2013['dest']==dest]=numID

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

    del carrier_2009
    del carrier_2010
    del carrier_2011
    del carrier_2012
    del carrier_2013
    del carrier_2014
    del dest_2009
    del dest_2010
    del dest_2011
    del dest_2012
    del dest_2013
    del dest_2014

    # Create training set and test set
    logfile = open("training_with_%d_procs_%d_trees.log"%(NPROCS,TREES_NUM),'w')

    train_y = [data_2009['delay']]
    train_x = [data_2009[cols]]

    train_y.append(data_2010['delay'])
    train_x.append(data_2010[cols])

    train_y.append(data_2011['delay'])
    train_x.append(data_2011[cols])

    train_y.append(data_2012['delay'])
    train_x.append(data_2012[cols])

    train_y.append(data_2013['delay'])
    train_x.append(data_2013[cols])

    test_y = data_2014['delay']
    test_x = data_2014[cols]

    train_x = pd.concat(train_x)
    train_y = pd.concat(train_y)
    print train_x.shape
    logfile.write ("Data Size %d %d\n"%(train_x.shape[0],train_x.shape[1]))

    for i in range(1,2):
        logfile.write ("Run Number %d\n"%(i))
        membefore = memory_usage_ps()
        forest = forest_generation(number_of_trees=10,features=train_x,delay=train_y,nprocs=NPROCS)
        memafter = memory_usage_ps()
    

        logfile.write ("Training Time: %f Memory Used by forest: %d\n"%(forest["time"],(memafter-membefore)))
        print "Training Time: ",forest["time"]," Memory Used by forest: ", (memafter-membefore), "kB"

        # Evaluate on test set
        predicted = predict(forest=forest["forest"],samples=test_x)

        # print results
        accur = accuracy_measure(predicted["predictions"],test_y)
        logfile.write("Accuracy: %f Absolute Number: %d. Prediction Time %f\n\n\n"%(accur['relative'],accur['abs'],predicted["time"]))
        print "Accuracy: ",accur['relative'],"Absolute Number: ",accur['abs'],"Prediction Time ",predicted["time"]
        del forest
        del predicted

    logfile.close()
