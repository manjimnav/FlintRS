#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np

import time

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import normalize, Normalizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


def calculate_metrics(y_test, y_pred, target, model_name):
    horizons = y_test.columns.values
  
    metrics = pd.DataFrame(classification_report(y_test.squeeze(), y_pred, output_dict=True))
    index = [np.array([model_name for i in range(4)]), metrics.index]
    metrics = metrics.set_index(index)

    return metrics


# In[ ]:


def execute_baseline(X_train, y_train, X_test, y_test, models, target):
    
    test_metrics_global = None
    train_metrics_global = None

    for name, model in models.items():
        
        best_model = None
        last_metric = float('-inf')
        print("Training "+name+"....")

        start_time = time.time()
        rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2)
        for train_index, test_index in rkf.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
            y_train_fold, y_test_fold = y_train.iloc[train_index,:], y_train.iloc[test_index,:]
            model.fit(X_train_fold,y_train_fold)
            metric = calculate_metrics(y_test_fold, model.predict(X_test_fold), target, name)["weighted avg"][2]
            if metric > last_metric:
                best_model = model
        print("--- %s seconds ---" % (time.time() - start_time))

        test_pred = best_model.predict(X_test)

        test_metrics = calculate_metrics(y_test, test_pred, target, name)
        
        if test_metrics_global is None:
            test_metrics_global = test_metrics

        else:
            test_metrics_global = test_metrics_global.append(test_metrics)
        
    return test_metrics_global
            


# In[ ]:


def execute(X_train, y_train, X_test, y_test, models, target="label"):

    print("============== "+target+" ==============")
    
    test_metrics= execute_baseline(X_train, y_train, X_test, y_test, models, target)
    test_metrics.to_pickle("drive/My Drive/Master/BigData/RecSysProduct/metrics/train_metrics_"+target)

    return test_metrics


# In[ ]:


X_train = pd.read_csv("drive/My Drive/Master/BigData/RecSysProduct/training_data/X_train.csv")
y_train = pd.read_csv("drive/My Drive/Master/BigData/RecSysProduct/training_data/y_train.csv")

X_test = pd.read_csv('drive/My Drive/Master/BigData/RecSysProduct/training_data/X_test.csv')
y_test = pd.read_csv('drive/My Drive/Master/BigData/RecSysProduct/training_data/y_test.csv')

X_train.drop(["idx", "man"], axis=1, inplace=True)
X_test.drop(["idx", "man"], axis=1, inplace=True)

y_train.drop(["idx"], axis=1, inplace=True)
y_test.drop(["idx"], axis=1, inplace=True)

X_train_sample_cut =int( X_train.shape[0]*0.1)
X_test_sample_cut =int(X_train_sample_cut*0.30)


# In[ ]:


# Decission tree
#distributions_dt = dict(max_depth=list(range(2,16)))
dt_model = DecisionTreeClassifier(max_depth=5)

# AdaBoost with Decission tree
#distributions_ada = {'estimator__base_estimator__max_depth':list(range(2,16))}
ada_cls_model = AdaBoostClassifier(DecisionTreeClassifier(),
                          n_estimators=300)

# Gradient boosting
grboost_model = GradientBoostingClassifier(n_estimators=300,loss='deviance', learning_rate=0.1,
                                                               max_depth=5)

# Gaussian process
kernel = 1.0 * RBF(1.0)
gausspr_model = GaussianProcessClassifier(kernel=kernel)

#Random forest
rand_forest_model = RandomForestClassifier(n_estimators=300,max_depth=5)

et_model = ExtraTreesClassifier(n_estimators=300)

neigh_model = KNeighborsClassifier(n_neighbors=5, weights="distance")

nn_model =  MLPClassifier(hidden_layer_sizes=(100,), learning_rate="adaptive", early_stopping=True)


models = {
    #"decision_tree_classifier": dt_model,
    #"adaboost_tree_classifier": ada_cls_model,
    #"gradient_boost": grboost_model,
    #"random_forest": rand_forest_model,
    "extra_trees": et_model,
    "neural_network": nn_model,
    "knn":neigh_model
}


# In[ ]:


et_model = ExtraTreesClassifier(n_estimators=300)
model = SelectFromModel(et_model).fit(X_train.iloc[:X_train_sample_cut,:], y_train.iloc[:X_train_sample_cut,:])


# In[49]:


get_ipython().run_cell_magic('time', '', '#selected_columns = X_train.columns[model.estimator_.feature_importances_ > model.threshold_]\n#X_train_selected = X_train[selected_columns]\n#X_test_selected = X_test[selected_columns]\n\nexecute(X_train.iloc[:X_train_sample_cut,:], y_train.iloc[:X_train_sample_cut,:], X_test.iloc[:X_test_sample_cut,:], y_test.iloc[:X_test_sample_cut,:], models)')


# In[50]:


pd.read_pickle('drive/My Drive/Master/BigData/RecSysProduct/metrics/train_metrics_label_no_selection')


# In[ ]:





# In[ ]:


nn_model =  MLPClassifier(hidden_layer_sizes=(100,), learning_rate="adaptive", early_stopping=True)

