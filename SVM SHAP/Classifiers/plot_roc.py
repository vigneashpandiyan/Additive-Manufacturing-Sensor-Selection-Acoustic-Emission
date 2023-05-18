#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def plot_roc(model,Featurespace,classspace,classes,Title1,Title2):
    
    
    # Binarize the output
    classspace = label_binarize(classspace, classes=classes)
    n_classes = classspace.shape[1]
    
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(66)
    #n_samples, n_features = Featurespace.shape
    
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(Featurespace, classspace, test_size=.25,
                                                        random_state=random_state)
    
    
    #y_score = model.decision_function(X_test)
    y_score = model.predict_proba(X_test)
    
    #print(y_score)
    #print(y_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    prec = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        prec[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
   
    lw = 2
    
 
    # Plot all ROC curves
    
    plt.figure(figsize = (8, 6),dpi=400)
    colors = cycle(['red', 'darkorange', 'green','blue','yellow'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(Title1,bbox_inches='tight',dpi=400)
    plt.show()
    
    # Plot all Precision-recall curves
    
    plt.figure(figsize = (8, 6),dpi=400)
    colors = cycle(['red', 'darkorange', 'green','blue','yellow'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], prec[i], color=color, lw=lw,
                 label='Precision-recall for class {0}  (area = {1:0.2f})'
                 ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(Title2,bbox_inches='tight',dpi=400)
    plt.show()
    
    

    # y_prob = classifier.predict_proba(X_test)
    
    # macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
    #                                   average="macro")
    # weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
    #                                      average="weighted")
    # macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
    #                                   average="macro")
    # weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
    #                                      average="weighted")
    # print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    #       "(weighted by prevalence)"
    #       .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    # print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    #       "(weighted by prevalence)"
    #       .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
