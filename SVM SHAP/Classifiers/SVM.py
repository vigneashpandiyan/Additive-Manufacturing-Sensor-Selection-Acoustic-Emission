#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split# implementing train-test-split
from numpy import mean
from numpy import std
import shap
from matplotlib.colors import ListedColormap
 
 
def SVM(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder):
    
    
    
    svc_model = SVC(kernel='rbf',probability=True, random_state=66)
    model=svc_model.fit(X_train,y_train)
    
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
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))  
    
    predictions = model.predict(X_test)
    print("SVM Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    graph_name1= 'SVM'+'_without normalization w/o Opt'
    graph_name2=  'SVM'
    
    graph_1= folder+'SVM'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= folder+'SVM'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        plt.figure(figsize = (20, 10),dpi=200)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                     display_labels=classes,
                                     cmap=plt.cm.Reds,xticks_rotation='vertical',
                                    normalize=normalize,values_format='0.2f')
        
        #disp.ax_.set_title(title)
        plt.title(title, size = 12)
        
        plt.savefig(graphname,bbox_inches='tight',dpi=200)
        plt.show()
        
    savemodel=  folder+'SVM'+'_model'+'.sav'
    joblib.dump(model, savemodel)
    
    X_test_, _, y_test_, _ = train_test_split(X_test, y_test, test_size=0.75, random_state=66)
    
    X_test_ = pd.DataFrame(X_test_)
    X_test_=X_test_.reset_index(drop=True)
    y_test_ = pd.DataFrame(y_test_)
    y_test_=y_test_.reset_index(drop=True)
    
    data = pd.concat([X_test_, y_test_],axis=1)
    minval = 30
    classes=np.unique(y_test_)
    print("Order of windows as per the class: ",classes)
    data = pd.concat([data[data.Categorical == cat].head(minval) for cat in classes])  
    data=data.sort_values(by="Categorical", ascending=True)
    data=data.reset_index(drop=True)
    
    X_test_=data.iloc[:,:-1]
    y_test_=data.iloc[:,-1]
    
    
    
    graphname= folder+'SHAP_'+'SVM'+'.png'
    explainer = shap.KernelExplainer(model.predict_proba, X_test_)
    shap_values = explainer.shap_values(X_test_)
    
    plt.figure(figsize = (5, 5),dpi=400)
    cmap =ListedColormap(['blue', 'green','red'])
    # shap.plots.bar(shap_values)
    shap.summary_plot(shap_values, X_test_.values, plot_type="bar", class_names= classes, feature_names = X_test_.columns,show=False, color=cmap)
    plt.ylabel('Features',fontsize=25)
    plt.xlabel('mean(|SHAP value|)\nAverage impact on model output magnitude',fontsize=20)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    # plt.legend('',frameon=False)
    plt.savefig(graphname,bbox_inches='tight',dpi=400)
    plt.show()
    
    
    
