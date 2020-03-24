
!pip install sklearn2pmml
!pip install --upgrade nyoka
!pip install keras-swa

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import randint as sp_randint

from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers

from google.colab import drive
drive.mount('/content/drive')

def calculate_metrics(y_test, y_pred, target, model_name):
    horizons = y_test.columns.values
  
    metrics = pd.DataFrame(classification_report(y_test.squeeze(), y_pred, output_dict=True))
    index = [np.array([model_name for i in range(4)]), metrics.index]
    metrics = metrics.set_index(index)

    return metrics

X_train = pd.read_csv("drive/My Drive/Master/BigData/RecSysProduct/training_data/X_train.csv")
y_train = pd.read_csv("drive/My Drive/Master/BigData/RecSysProduct/training_data/y_train.csv")

X_test = pd.read_csv('drive/My Drive/Master/BigData/RecSysProduct/training_data/X_test.csv')
y_test = pd.read_csv('drive/My Drive/Master/BigData/RecSysProduct/training_data/y_test.csv')

X_train.drop(["idx", "man"], axis=1, inplace=True)
X_test.drop(["idx", "man"], axis=1, inplace=True)

y_train.drop(["idx"], axis=1, inplace=True)
y_test.drop(["idx"], axis=1, inplace=True)


parameter_space = {
    'hidden_layer_sizes': [(sp_randint.rvs(20,500,1),sp_randint.rvs(2,500,1),), 
                                          (sp_randint.rvs(10,500,1),),
                            (sp_randint.rvs(10,1000,1), sp_randint.rvs(10,500,1), sp_randint.rvs(10,500,1)),
                            (sp_randint.rvs(10,500,1), sp_randint.rvs(10,500,1), sp_randint.rvs(10,500,1), sp_randint.rvs(10,500,1))],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    'learning_rate_init': [0.1, 0.01, 0.05, 0.001, 0.2],
    'max_iter': [200, 400, 600, 800, 1000]
}


nn_model =  MLPClassifier(early_stopping=True)


clf = RandomizedSearchCV(nn_model, parameter_space, n_jobs=-1, cv=2, n_iter=5)

%%time
search = clf.fit(X_train, y_train)

search.best_params_

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

pipeline_obj = Pipeline([
    ('scaler', MinMaxScaler()),
    ('nn',MLPClassifier(activation="tanh", alpha=0.05, hidden_layer_sizes=(63, 15), learning_rate="adaptive", learning_rate_init=0.05, max_iter=200, solver="sgd", early_stopping=True))
])
pipeline_obj.fit(X_train, y_train)


from nyoka import skl_to_pmml
skl_to_pmml(pipeline_obj,X_train.columns,'label',"nn_pmml.pmml")

calculate_metrics(y_test, pipeline_obj.predict(X_test), "label", "NN")
