#support vector machine
#dataset ALL and AML


#necesario instalar las librerias en la consola
#librerias necesarias
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
#%matplotlib inline

#cargar datos 
ruta_train='C:\\Users\\Elsa\\Documents\\BIOINFORMÁTICA\\PROYECTO\\Datos\\ORIGINALES\\data_set_ALL_AML_train.csv'

ruta_y='C:\\Users\\Elsa\\Documents\\BIOINFORMÁTICA\\PROYECTO\\Datos\\ORIGINALES\\actual.csv'

ruta_test='C:\\Users\\Elsa\\Documents\\BIOINFORMÁTICA\\PROYECTO\\Datos\\ORIGINALES\\data_set_ALL_AML_independent.csv'

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

train=train.astype('float') # es X

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

atributos1



#Eliminacion recursiba de atributos

modelo = ExtraTreesClassifier()

era = RFE(modelo, 100)  # número de atributos a seleccionar

era = era.fit(train, ytrain['cancer'])

atrib2 = era.support_

atributos2 = [columnas[i] for i in list(atrib2.nonzero()[0])]

atributos2



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





clf_gini = DecisionTreeClassifier(criterion = 'gini', random_state=100)

clf_gini.fit(train, ytrain)



ypred=clf_gini.predict(test)



#Random forest











#Super Vector Machine
#info  -->https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/


from sklearn.svm import SCV
scvclassifier = SCV(kernel='linear')
scvclassifier.fit(train, ytrain['cancer'])

#predicciones
y_pred = scvclassifier.predict(test)


#evaluación del algoritmo
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ytest, y_pred))
print(classification_report(y_test, y_pred))
	
