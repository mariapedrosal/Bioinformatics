#Es necesario tener instaladas los siguientes paquetes de python (pip install paquete en simbolo del sistema)

#import scipy
import numpy as np
#import matplotlib
import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
##Importar los metodos que vamos a utilizar
from sklearn.metrics import classification_report
#from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt
#from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
     
import pandas as pd
ruta_train='C:\\Users\\Maria\\Documents\\Master\\Bioinformatica\\Trabajo\\Originales\\data_set_ALL_AML_train.csv'
ruta_y='C:\\Users\\Maria\\Documents\\Master\\Bioinformatica\\Trabajo\\Originales\\actual.csv'
ruta_test='C:\\Users\\Maria\\Documents\\Master\\Bioinformatica\\Trabajo\\Originales\\data_set_ALL_AML_independent.csv'
train=pd.read_csv(ruta_train)
test=pd.read_csv(ruta_test)
y=pd.read_csv(ruta_y)

#Transponer
train=train.T
test=test.T


#Quitar los call
for fila in train.index:
    if 'call' in fila:
        train=train.drop(fila)

for fila in test.index:
    if 'call' in fila:
        test=test.drop(fila)

#Poner como nobre de columnas el gene accession

columnastrain=train.loc['Gene Accession Number']
train=train[2:]
train.columns=columnastrain

columnastest=test.loc['Gene Accession Number']
test=test[2:]
test.columns=columnastest

#Pasar a numerico
train=train.astype('float')
test=test.astype('float')
train.index=train.index.astype(int)
test.index=test.index.astype(int)
train=train.sort_index()
test=test.sort_index()
#COdificar la variable respuesta (categorica) como 0 y 1
y.index=y['patient']
y['cancer']=np.where(y.cancer=='ALL', 1, 0) #ALL es la categoria 1, AML es la 0
y.groupby('cancer').size() #Hay 47 AML y 25 ALL, por lo que esta desbalancedada
ytrain=y[:len(train)]
ytest=y[len(train):]

#Hay demasiados features, hay que quitar los que sobran

# Aplicando el algoritmo univariante de prueba F.
k = 100  # número de atributos a seleccionar
columnas = list(train.columns.values)
seleccionadas = SelectKBest(f_classif, k=k).fit(train, ytrain['cancer'])
atrib = seleccionadas.get_support()
atributos1 = [columnas[i] for i in list(atrib.nonzero()[0])]

#Eliminacion recursiba de atributos
modelo = ExtraTreesClassifier()
era = RFE(modelo, 100)  # número de atributos a seleccionar
era = era.fit(train, ytrain['cancer'])
atrib2 = era.support_
atributos2 = [columnas[i] for i in list(atrib2.nonzero()[0])]

#Se combinan los tributos elegidos en una lista
atribselec=list(set(atributos1)|set(atributos2))

trainred=pd.DataFrame()
for i in atribselec:
    trainred[i]=train[i]

testred=pd.DataFrame()
for i in atribselec:
    testred[i]=test[i]
#Aun asi pueden ser demasiados atributos, por lo que se realiza un analisis de 
#componentes principales
scaler=StandardScaler()
scaler.fit(trainred)
train=scaler.transform(trainred)
test=scaler.transform(testred)

pca=PCA(.95) #El algoritmo elige el numero minimo de componentes de manera que  aun se retiene el 95% de la varianza
pca.fit(trainred)

train=pca.transform(trainred)
test=pca.transform(testred)

##ÏCreamo un arbol de decision
#cv = KFold(n_splits=10)
#accuracies = list()
#max_attributes = len(list(ytrain))
#depth_range = range(1, max_attributes + 1)

#Buscar los mejores parametros por cross validation
split_range=list(range(2,15))
prof_range=list(range(2, 10))

param_grid={'min_samples_split':split_range, 'max_depth':prof_range}
clf_gini = DecisionTreeClassifier(criterion = 'gini', random_state=100)
grid_dt=GridSearchCV(clf_gini, param_grid, scoring='accuracy')
grid_dt.fit(train, ytrain['cancer'])
print(grid_dt.best_score_)
print(grid_dt.best_params_)

mejor_clf_gini=grid_dt.best_estimator_


ypred=mejor_clf_gini.predict(test)
print(mejor_clf_gini.predict_proba(test))
print(pd.crosstab(ytest['cancer'],ypred, rownames=['Actual diagnosis'], colnames=['Predicted diagnosis']))
ypred_t=mejor_clf_gini.predict(train)
print(pd.crosstab(ytrain['cancer'],ypred_t, rownames=['Actual diagnosis'], colnames=['Predicted diagnosis']))
print(classification_report(ytest['cancer'], ypred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(ytest['cancer'], ypred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
#Visualización
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import graphviz
#dot_data = tree.export_graphviz(mejor_clf_gini, out_file=None, filled=True, rounded=True, special_characters=True)  
#graph = graphviz.Source(dot_data)  
#graph

#Random forest
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_clf=RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf_clf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=100, n_jobs = -1)
rf_random.fit(train, ytrain['cancer'])



print(rf_random.best_params_)

param_grid = {
    'bootstrap': [True],
    'max_depth': [None],
    'max_features': ['auto'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [500, 1000, 2000]
}

grid_rf=GridSearchCV(rf_clf, param_grid, scoring='accuracy', cv=3)
grid_rf.fit(train, ytrain['cancer'])
print(grid_rf.best_score_)
print(grid_rf.best_params_)

mejor_rf_clf=grid_rf.best_estimator_


ypred_rf=mejor_rf_clf.predict(test)
print(mejor_rf_clf.predict_proba(test))
print(pd.crosstab(ytest['cancer'],ypred_rf, rownames=['Actual diagnosis'], colnames=['Predicted diagnosis']))
ypred_t_rf=mejor_rf_clf.predict(train)
print(pd.crosstab(ytrain['cancer'],ypred_t_rf, rownames=['Actual diagnosis'], colnames=['Predicted diagnosis']))
print(classification_report(ytest['cancer'], ypred_rf))

false_positive_rate, true_positive_rate, thresholds = roc_curve(ytest['cancer'], ypred_rf)
roc_auc_rf = auc(false_positive_rate, true_positive_rate)
roc_auc_rf

#Neural network
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(train, ytrain['cancer'])
nn_pred=mlp.predict(test)
print(pd.crosstab(ytest['cancer'],nn_pred, rownames=['Actual diagnosis'], colnames=['Predicted diagnosis']))
print(classification_report(ytest['cancer'], nn_pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(ytest['cancer'], nn_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
