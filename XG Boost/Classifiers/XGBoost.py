# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:29:36 2023

@author: srpv
"""

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from xgboost import XGBClassifier


#%%

def XGBoost(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder,dataset_name):
    

    model = XGBClassifier(max_depth=4,booster='gbtree')
    model.fit(X_train,y_train)
    
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    y_test.columns = ['Categorical']
    data = pd.concat([X_test, y_test], axis=1)
    minval = min(data.Categorical.value_counts())
    print("windows of the class: ",minval)
    
    classes=np.unique(y_train)
    print("Order of windows as per the class: ",classes)
    data_1 = pd.concat([data[data.Categorical == cat].head(minval) for cat in classes])  
    print("The dataset is well balanced: ",data_1.Categorical.value_counts())
    
    X_test=data_1.iloc[:,:-1]
    y_test=data_1.iloc[:,-1]
    
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model,X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))  
    
    
    predictions = model.predict(X_test)
    print("XGBoost Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    

    graph_name1= 'XGBoost'+'_without normalization w/o Opt'
    graph_name2=  'XGBoost'
    
    graph_1=folder+ 'XGBoost'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= folder+'XGBoost'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    # plt.bar(range(len(model.feature_importances_)), model.feature_importances_)

    print(model.get_xgb_params())
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        plt.figure(figsize = (20, 10),dpi=400)
        
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                      display_labels=classes,
                                      cmap=plt.cm.pink,xticks_rotation='vertical',
                                    normalize=normalize,values_format='0.2f')
        plt.title(title, size = 12)
        
        plt.savefig(graphname,bbox_inches='tight',dpi=400)
    savemodel= folder+ 'XGBoost'+'_model'+'.sav'    
    joblib.dump(model, savemodel)
    
    sorted_idx = model.feature_importances_.argsort()
    
    plt.figure(figsize = (4, 4),dpi=400)
    plt.barh(X_test.columns[sorted_idx], model.feature_importances_[sorted_idx],color="limegreen")
    plt.title('Dataset '+ dataset_name,fontsize=15)
    plt.xlabel('F score (Normalized)',fontsize=15)
    plt.ylabel('Features',fontsize=15)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=12)
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    graphname=folder+'Feature_XGBoost.png'
    plt.savefig(graphname,bbox_inches='tight',dpi=400)
    plt.show()
