#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This programm takes the two-principal component arrays result from 
"hist3.py" programm and to apply learning models to compare each other

Created on Fri Apr 10 18:57:29 2020

@author: erickmfs
"""
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
#import random
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from scipy import interp
from matplotlib import pyplot
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

ds_n = "/home/erickmfs/ai_apps/neuromatia_diabetica_models/dataset/DB_Pmax_sinID_389.csv"
kfold = 5 # no. of folds (better to have this at the start of the code)

################# - CREATING MODELS ARE GOING TO ASSESS - #####################
# Decision Treee Classifier
# Random Forest Classifier
# K-Nearest Neighbor 
# Support Vector Machine
###############################################################################


def gen_skf_indices(X,y,n_samples,kfold):
    skf = StratifiedKFold(kfold)
    skf.get_n_splits(X,y)
        
    # Stratified KFold: This first divides the data into k folds. Then it also makes sure that the distribution of the data in each fold follows the original input distribution 
    skfinds = [None] * n_samples
    
    p = 0
    for idx in skf.split(X,y):
        #print(idx)
        skfinds[p] = idx
        p += 1
    
    return skfinds

def params_assess(model4grid,param_grid,X_train,y_train,values,index,columns,name_model):
    print("****************** GRID SEARCH CROSS VALIDATION PROCESS **********************")
    grid = GridSearchCV(model4grid,param_grid,refit=True)#,verbose=3)
    grid.fit(X_train,y_train)
    
    print("Best parameters: " + str(grid.best_params_))
    print("Best estimator: " + str(grid.best_estimator_))
    print("Best score: " + str(grid.best_score_))
    
    grid_predictions = grid.predict(X_test)
    print(confusion_matrix(y_test,grid_predictions))
    print(classification_report(y_test,grid_predictions))
    
    pyplot.figure(figsize=(10,6))
    pyplot.title("Grid search cross validation for: " + name_model + " model")
    pvt = pd.pivot_table(pd.DataFrame(grid.cv_results_), values=values, index=index, columns=columns)
    ax = sns.heatmap(pvt)
    return grid.best_params_

def train_model(model, X, y, name_model):
    #This time the program is going to train the model based on KCV taking into
    #consideration ROC index
    print("****************** TRAINING PROCESS INITIATED **********************")
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(n_splits=kfold, random_state = 0, shuffle = True)
    fig, ax = pyplot.subplots(figsize=(10,6))
    for i, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train])
        viz = plot_roc_curve(model, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC validation for: " + name_model + " model")
    ax.legend(loc="lower right")
    pyplot.show()
        
    #all_accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=kfold)
    #print("cross_val_score class results:")
    #print(all_accuracies)
    #print("mean: " + str(all_accuracies.mean()))
    #print("std: " + str(all_accuracies.std()))

def customize_train_model(X,y,kfold,model,skfinds,n_samples):
    print("Starting up training process")
    conf_mat = np.zeros((2,2))
    
    clf = []
    clf = model
    
    #print(y)
    
    for i in range(kfold):
        train_indices = skfinds[i][0]
        test_indices = skfinds[i][1]
        
        
        X_train = X[train_indices,:]
        y_train = y[train_indices]
        X_test = X[test_indices,:]
        y_test = y[test_indices]
        
        #print(X_train.shape)
        #print(X_train)
        #print(y_test.shape)
        #print(y_test)
        
        #training
        tic = time.time()
        clf.fit(X_train,y_train)
        toc = time.time()
        print("training time = ", str(toc-tic))
        
        #testing
        y_predict = []
        tic = time.time()
        y_predict = clf.predict(X_test) #output is labels and not indices
        toc = time.time()
        print("testing time = ", str(toc-tic))
        
        #compute confusion matrix
        cm = []
        cm = confusion_matrix(y_test,y_predict)
        print(cm)
        conf_mat = conf_mat + cm
        print(conf_mat)
    conf_mat = conf_mat.T # since rows and  cols are interchanged
    avg_acc = np.trace(conf_mat)/n_samples
    print(avg_acc)
    #conf_mat_norm = conf_mat/n_samples # Normalizing the confusion matrix
    
    return conf_mat, avg_acc, clf  #, conf_mat_norm

def get_pc(path): #get principal componentes after hist3.py analysis
    pdata = pd.read_csv(path)
    #data = pdata.pop("Dx")
    data_target = pdata.pop("Dx")
    data_features = pdata
    # Standardizing the features
    sdf = StandardScaler().fit_transform(data_features)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(sdf)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    
    return np.array(principalDf), data_target, len(principalDf)
  
def testing_model(model,X_test,y_test, name_model):
    
    print("****************** TESTING PROCESS INITIATED **********************")
    predictions = model.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    
    auc = roc_auc_score(y_test, predictions)
    print('AUC: %.3f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    pyplot.figure(figsize=(10,6))
    pyplot.title("Final ROC validation for: " + name_model + " model")
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.show()

def plot_pca(data,pxtrain,nxtrain,pxtest,nxtest,dattitulo,ptrtitulo,ntrtitulo,ptetitulo,ntetitulo):
    
    fig, axes= pyplot.subplots(nrows=3, ncols=2,figsize=(15,10))   
    
    labels = ["","PCA1","PCA2"]
    x = np.arange(len(labels))
    
    axes[0,0].set_ylabel('Scores')
    axes[0,0].set_title(dattitulo)
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(labels)
    axes[0,0].legend()
    if len(data) > 0:
        collectn_1 = data[:,0]
        collectn_2 = data[:,1]
    else: #in case there is no data to show
        collectn_1 = np.array([0])
        collectn_2 = np.array([0])
    ## combine these different collections into a list
    data_to_plot = [collectn_1, collectn_2]
    axes[0,0].violinplot(data_to_plot)
        
    axes[1,0].set_ylabel('Scores')
    axes[1,0].set_title(ptrtitulo)
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(labels)
    axes[1,0].legend()
    if len(pxtrain) > 0:
        collectn_1 = pxtrain[:,0]
        collectn_2 = pxtrain[:,1]
    else: #in case there is no data to show
        collectn_1 = np.array([0])
        collectn_2 = np.array([0])
    ## combine these different collections into a list
    data_to_plot = [collectn_1, collectn_2]
    axes[1,0].violinplot(data_to_plot)
    
    axes[1,1].set_ylabel('Scores')
    axes[1,1].set_title(ntrtitulo)
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(labels)
    axes[1,1].legend()
    if len(nxtrain) > 0:
        collectn_1 = nxtrain[:,0]
        collectn_2 = nxtrain[:,1]
    else: #in case there is no data to show
        collectn_1 = np.array([0])
        collectn_2 = np.array([0])
    ## combine these different collections into a list
    data_to_plot = [collectn_1, collectn_2]
    axes[1,1].violinplot(data_to_plot)
    
    axes[2,0].set_ylabel('Scores')
    axes[2,0].set_title(ptetitulo)
    axes[2,0].set_xticks(x)
    axes[2,0].set_xticklabels(labels)
    axes[2,0].legend()
    if len(pxtest) > 0:
        collectn_1 = pxtest[:,0]
        collectn_2 = pxtest[:,1]
    else: #in case there is no data to show
        collectn_1 = np.array([0])
        collectn_2 = np.array([0])
    ## combine these different collections into a list
    data_to_plot = [collectn_1, collectn_2]
    axes[2,0].violinplot(data_to_plot)
    
    axes[2,1].set_ylabel('Scores')
    axes[2,1].set_title(ntetitulo)
    axes[2,1].set_xticks(x)
    axes[2,1].set_xticklabels(labels)
    axes[2,1].legend()
    if len(nxtest) > 0:
        collectn_1 = nxtest[:,0]
        collectn_2 = nxtest[:,1]
    else: #in case there is no data to show
        collectn_1 = np.array([0])
        collectn_2 = np.array([0])
    ## combine these different collections into a list
    data_to_plot = [collectn_1, collectn_2]
    axes[2,1].violinplot(data_to_plot)
    
    pyplot.tight_layout()
    pyplot.show()

def extract_p_n(data,y):
    y = np.array(y)
    pdata = []
    ndata = []
    
    for i in range(len(data)):
        if y[i] == 1:
            pdata.append(data[i,:])
        else:
            ndata.append(data[i,:])
    pdata = np.array(pdata)
    ndata = np.array(ndata)
    
    return pdata, ndata

def get_charts_ga_pn(data, X_train, X_test, y_train, y_test, dattitulo): #meeting the negative and positive part of x train independently 
    pxtrain, nxtrain = extract_p_n(X_train,y_train)
    pxtest, nxtest = extract_p_n(X_test,y_test)
    
    #dattitulo = 'Scores by PCAs'
    ptrtitulo = 'Scores by PCAs with positive X_train (shuffle:y/stratify:y)' 
    ntrtitulo = 'Scores by PCAs with negative X_train (shuffle:y/stratify:y)'
    ptetitulo = 'Scores by PCAs with positive X_test (shuffle:y/stratify:y)' 
    ntetitulo = 'Scores by PCAs with negative X_test (shuffle:y/stratify:y)'
    
    plot_pca(data,pxtrain,nxtrain,pxtest,nxtest,dattitulo,ptrtitulo,ntrtitulo,ptetitulo,ntetitulo)

def prob_dist_an(principalDf, data_target, test_size_array):
    
    for i in range(len(np.transpose(test_size_array))):
        X_train, X_test, y_train, y_test = train_test_split(principalDf, data_target, test_size=test_size_array[0,i], random_state=0, shuffle=True, stratify=data_target)
        dattitulo = 'Scores by PCAs - test size: [' + str(test_size_array[0,i]) + ']'
        get_charts_ga_pn(principalDf, X_train, X_test, y_train, y_test, dattitulo)
    
#--------------------------- get necessary data--------------------------------
principalDf, data_target, n_samples = get_pc(ds_n)

test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(principalDf, data_target, test_size=test_size, random_state=0, shuffle=True, stratify=data_target)
y_train = np.array(y_train)
y_test = np.array(y_test)

#skfinds = []
#skfinds = gen_skf_indices(X_train,y_train,n_samples,kfold)

#For analysis purposes 
prob_dist_analysis = 0
if prob_dist_analysis == 1:
    test_size_array = np.array([[0.20, 0.25, 0.30, 0.35, 0.40, 0.45]])
    prob_dist_an(principalDf, data_target, test_size_array)
    
applymodels = 1 #key for models to be shown
if applymodels == 1:
#------------------------------TRAINING MODELS---------------------------------
#------------------------------------------------------------------------------
#                        DECISION TREE CLASSIFIER
#                        ------------------------
    name_model = "Decision tree classifier"
    param_grid = {
            'max_depth': [3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'min_samples_split': [2,3,4,5,6,7,8],
            'criterion': ['entropy']
            }
    values = 'mean_test_score'
    index = 'param_max_depth'
    columns = 'param_min_samples_split'
    best_params = params_assess(DecisionTreeClassifier(), param_grid, X_train, y_train, values, index, columns, name_model)
    
    dtree = []
    dtree = DecisionTreeClassifier(criterion='entropy',max_depth=best_params['max_depth'],min_samples_split=best_params['min_samples_split'],random_state=0)    
    train_model(dtree,X_train,y_train, name_model)
    
    testing_model(dtree, X_test, y_test, name_model)

#                         RANDOM FOREST CLASSIFIER
#                         ------------------------
    name_model = "Random forest classifier"
    param_grid = {
            'max_depth': [3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'min_samples_split': [2,3,4,5,6,7,8],
            'criterion': ['entropy']
            }  
    values = 'mean_test_score'
    index = 'param_max_depth'
    columns = 'param_min_samples_split'
    
    #testing_model(rfc,RandomForestClassifier(),X_test,y_test,param_grid,values,index,columns)
    best_params = params_assess(RandomForestClassifier(), param_grid, X_train, y_train, values, index, columns, name_model)

    rfc = []
    rfc = RandomForestClassifier(criterion='entropy',max_depth=best_params['max_depth'],min_samples_split=best_params['min_samples_split'],random_state=0)
    #conf_mat, avg_acc, rfc = train_model(X_train,y_train,kfold,rfc,skfinds,n_samples)
    train_model(rfc,X_train,y_train,name_model)  
    
    testing_model(rfc, X_test, y_test, name_model)

#                        KNN (k-NEAREST NEIGHBORS)
#                        ---------------------------    
    name_model = "K-nearest neighbors"
    param_grid = {
            #'leaf_Size': [30,31,32,33,34,35,36], #,27,28,29,30,31,32,33
            'n_neighbors': [2,3,4,5,6,7,8],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
    values = 'mean_test_score'
    index = 'param_algorithm'
    columns = 'param_n_neighbors'
    #testing_model(knn,KNeighborsClassifier(),X_test,y_test,param_grid,values,index,columns)
    best_params = params_assess(KNeighborsClassifier(), param_grid, X_train, y_train, values, index, columns, name_model)

    #conf_mat, avg_acc, knn = train_model(X_train,y_train,kfold,knn,skfinds,n_samples)
    knn = []
    knn = KNeighborsClassifier(algorithm = best_params['algorithm'], n_neighbors=best_params['n_neighbors'])
    train_model(knn,X_train,y_train, name_model)
    
    testing_model(knn, X_test, y_test, name_model)

#                         SUPPORT VECTOR MACHINE
#                         ----------------------
    name_model = "Support vector machine"
    param_grid = {
            'C': [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1, 10, 100, 1000,10000], 
            'gamma': [1000000,100000,10000,1000,100,10,1,0.1,0.01,0.001,0.0001], 
            #'kernel': ['linear','rbf','poli','sigmoid','precomputed']
            'kernel': ['rbf']
            }     
    values = 'mean_test_score'
    index = 'param_C'
    columns = 'param_gamma'
    #testing_model(svm,SVC(),X_test,y_test,param_grid,values,index,columns)
    best_params = params_assess(SVC(), param_grid, X_train, y_train, values, index, columns, name_model)
    
    svm = []
    svm = SVC(C=best_params['C'], cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma=best_params['gamma'], kernel=best_params['kernel'],
          max_iter=-1, probability=False, random_state=None, shrinking=True,
          tol=0.001, verbose=False)
    #conf_mat, avg_acc, svm = train_model(X_train,y_train,kfold,svm,skfinds,n_samples)
    train_model(svm,X_train,y_train,name_model)
        
    testing_model(svm, X_test, y_test, name_model)

