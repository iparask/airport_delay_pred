import warnings
warnings.filterwarnings('ignore')

import sys
import random
import numpy as np
from datetime import datetime
from sklearn import linear_model, cross_validation, metrics, svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os

def read_csv_from_dir(path, cols, col_types=None):
    pieces = []
    for f in os.listdir(path):
        if f[0] != '.':
            fhandle = open(os.path.join(path,f),'r')
            pieces.append(pd.read_csv(fhandle, names=cols, dtype=col_types))
            fhandle.close()
    return pd.concat(pieces, ignore_index=True)

if __name__ == '__main__':

    # read files
    cols = ['delay', 'month', 'day', 'dow', 'hour', 'distance', 'carrier', 'dest', 'days_from_holiday']
    col_types = {'delay': int, 'month': int, 'day': int, 'dow': int, 'hour': int, 'distance': int, 
	   'carrier': str, 'dest': str, 'days_from_holiday': int}
    data_2002 = read_csv_from_dir('../data/ord_2002_1', cols, col_types)
    data_2003 = read_csv_from_dir('../data/ord_2003_1', cols, col_types)
    data_2004 = read_csv_from_dir('../data/ord_2004_1', cols, col_types)
    data_2005 = read_csv_from_dir('../data/ord_2005_1', cols, col_types)
    data_2006 = read_csv_from_dir('../data/ord_2006_1', cols, col_types)
    data_2007 = read_csv_from_dir('../data/ord_2007_1', cols, col_types)
    data_2008 = read_csv_from_dir('../data/ord_2008_1', cols, col_types)

    # Create training set and test set
    cols = ['month', 'day', 'dow', 'hour', 'distance', 'days_from_holiday']
    train_y = data_2002['delay'] >= 15
    train = data_2003['delay'] >= 15
    train_y = [train_y]#, train]
    train = data_2004['delay'] >= 15
    train_y.append(train)
    train = data_2005['delay'] >= 15
    train_y.append(train)
    train = data_2006['delay'] >= 15
    train_y.append(train)
    train = data_2007['delay'] >= 15
    train_y.append(train)
    train_x = [data_2002[cols],data_2003[cols],data_2004[cols],data_2005[cols],data_2006[cols],data_2007[cols]]
    test_y = data_2008['delay'] >= 15
    test_x = data_2008[cols]

    train_x = pd.concat(train_x)
    train_y = pd.concat(train_y)
    print train_x.shape


    start = datetime.now()
    clf_rf = RandomForestClassifier(n_estimators=50, n_jobs=4)
    clf_rf.fit(train_x, train_y)
    total_time = (datetime.now()-start).total_seconds()
    print total_time
    # Evaluate on test set
    pr = clf_rf.predict(test_x)

    # print results
    cm = confusion_matrix(test_y, pr)
    print("Confusion matrix")
    print(pd.DataFrame(cm))
    report_svm = precision_recall_fscore_support(list(test_y), list(pr), average='micro')
    print "\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
        (report_svm[0], report_svm[1], report_svm[2], accuracy_score(list(test_y), list(pr)))