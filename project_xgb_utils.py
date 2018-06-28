import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics

def train_model(param, d_train, num_boost_round):
    start = time.time()
    bst = xgb.train(param, d_train, num_boost_round)
    end = time.time()
    fit_time = end - start
    print('Time elapsed (Training): %.4f s' % fit_time)

    return bst, fit_time


def evaluate(bst, d_train, y_train, d_valid, y_valid, plot=True):
    # Plotting ROCAUC and PRAUC
    f, (plt1, plt2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))        
    plt1.set_title('ROC Curve')
    plt1.set_xlabel('FPR')
    plt1.set_ylabel('TPR')
    plt2.set_title('PR Curve')
    plt2.set_xlabel('Precision')
    plt2.set_ylabel('Recall')

    start = time.time()

    # Training set
    y_train_predicted = bst.predict(d_train)
    fpr, tpr, thresholds = metrics.roc_curve(y_train.values, y_train_predicted)
    precision, recall, thresholds = metrics.precision_recall_curve(y_train.values, y_train_predicted)
    plt1.scatter(fpr, tpr, color='b')    
    plt2.scatter(precision, recall,color='b')
    print('Accuracy Score (Training): %f' % metrics.accuracy_score(y_train.values, [round(value) for value in y_train_predicted]))
    print('ROCAUC Score (Training): %f' % metrics.roc_auc_score(y_train.values, y_train_predicted))
    print('PRAUC Score (Training): %f' % metrics.auc(precision, recall, reorder=True))

    # Validation set
    y_valid_predicted = bst.predict(d_valid)
    fpr, tpr, thresholds = metrics.roc_curve(y_valid.values, y_valid_predicted)
    precision, recall, thresholds = metrics.precision_recall_curve(y_valid.values, y_valid_predicted)
    plt1.scatter(fpr, tpr, color='r')    
    plt2.scatter(precision, recall,color='r')
    acc_score = metrics.accuracy_score(y_valid.values, [round(value) for value in y_valid_predicted])
    print('Accuracy Score (Validation): %f' % acc_score)
    rocauc_score = metrics.roc_auc_score(y_valid.values, y_valid_predicted)
    print('ROCAUC Score (Validation): %f' % rocauc_score)
    prauc_score = metrics.auc(precision, recall, reorder=True)
    print('PRAUC Score (Validation): %f' % prauc_score)

    end = time.time()
    eval_time = end - start
    print('Time elapsed (Evaluation): %.4f s' % eval_time)

    if plot:
            plt.show()
    plt.close()

    return acc_score, rocauc_score, prauc_score, eval_time