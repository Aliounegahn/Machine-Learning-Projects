# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 08:19:54 2018


@author: Gahn Alioune
"""
import pandas as pa 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#fonction split
def split(dtf,p):
    msk = np.random.rand(len(dtf)) <= (p/100)
    train = df[msk]
    test = df[~msk]
    return([train,test]) 
#fonction prediction
def prediction(labels,pred):
     n = 0
     for i in range(len(labels)) :
          if((pred[i]) == labels.values[i]):
              n = n +1
              
     return ((1 - (n/len(labels)))*100)   
#chargement données  
col_names = ["age", "workclass", "fnlwgt", "education", "education_num", "maritalStatus", "occupation", "relationship", "race", "sex", "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry", "isUpper"];
col_cats = ["workclass", "education", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry", "isUpper"];

dtypes = {
    col: 'category'
    for col in col_cats
}

df=pa.read_csv('/home/sid2018-1/Téléchargements/adult.data',names=col_names, dtype=dtypes) 
print(df);
print(df.shape) 

print(df.info())
#Choix des attributs
df = df.drop(['age','occupation','maritalStatus'],axis=1)
print(df.head())
#Conversion de données 
df=pa.get_dummies(df,columns=['isUpper','workclass','education','relationship','race','sex','nativeCountry'])

#Remplacement des NA par la moyenne des valeurs dans la colonne
#values = {'workclass': df['workclass'].mean(), 'fnlwgt': df['fnlwgt'].mean(), 'education': df['education'].mean(), 'education_num': df['education_num'].mean(), 'relationship': df['relationship'].mean(), 'Sex_female': df['Sex_female'].mean(), 'Sex_male': df['Sex_male'].mean() ,'Embarked_C':df['Embarked_C'].mean(), 'Embarked_Q':df['Embarked_Q'].mean(), 'Embarked_S':df['Embarked_S'].mean()}

#df = df.fillna(value=values)
#print(df)


#appel fonction split 

l = split(df,80) 

#Construction des vecteurs train et test  
mat= l[0]
Xtrain=mat.drop(['isUpper_ >50K'],axis=1)
Ytrain=mat['isUpper_ >50K']
mat1= l[1]
Xtest=mat1.drop(['isUpper_ >50K'],axis=1)
Ytest=mat1['isUpper_ >50K']

Ytrain = Ytrain.ravel()
Ytest.ravel()

#gaussian
clf_gaussian = GaussianNB()
clf_gaussian.fit(Xtrain, Ytrain)
ypred_gaussian = clf_gaussian.predict(Xtest)
taux_gaussian = prediction(Ytest,ypred_gaussian)

#Regression Logistique
clf_lmlr = linear_model.LogisticRegression()
clf_lmlr.fit(Xtrain, Ytrain)
ypred_lmlr = clf_lmlr.predict(Xtest)
taux_lmlr = prediction(Ytest,ypred_lmlr)



#kneighbors
clf_kneighbors= KNeighborsClassifier()
clf_kneighbors.fit(Xtrain, Ytrain)
ypred_kneighbors = clf_kneighbors.predict(Xtest)
taux_kneighbors = prediction(Ytest,ypred_kneighbors)

#Nearest Centroid
clf_Nearest = NearestCentroid()
clf_Nearest.fit(Xtrain, Ytrain)
ypred_Nearest = clf_Nearest.predict(Xtest)
taux_Nearest = prediction(Ytest,ypred_Nearest)

## PARAMETRES OPTIMAUX 
#Kneighbors
scaler =StandardScaler().fit(Xtrain)
Xtrain1 = scaler.transform(Xtrain)
scaler =StandardScaler().fit(Xtest)
Xtest1 = scaler.transform(Xtest)
k_range = (1,10)
param_grid = dict(n_neighbors= k_range)
grid= GridSearchCV(KNeighborsClassifier(), param_grid,cv=5) 
grid.fit(Xtrain1, Ytrain)
knn = KNeighborsClassifier(n_neighbors= grid.best_params_["n_neighbors"]) 
knn.fit(Xtrain1, Ytrain)
ypred_knn = knn.predict(Xtest1)
taux_knn_opt = prediction(Ytest, ypred_knn)
# Gaussienne
taux_gaussienne = []
priors = np.linspace(0.001,1,1000, False)
for i in range(0, len(priors)):
    clf_gaussian.set_params(priors  = [priors[i], 1-priors[i]])
    clf_gaussian.fit(Xtrain, Ytrain)
    pred = clf_gaussian.predict(Xtest)
    taux_gaussienne = prediction(Ytest, pred)

taux_opt_gauss = priors[np.argmin(taux_gaussienne)]


    
    
#VALIDATION CROISEE 
score_gaussian = cross_val_score(clf_gaussian, Xtrain, Ytrain, cv=5)
score_Nearest = cross_val_score(clf_Nearest, Xtrain, Ytrain, cv=5)
score_kneighbors = cross_val_score(knn, Xtrain1, Ytrain, cv=5)
score_lmlr = cross_val_score(clf_lmlr, Xtrain1, Ytrain, cv=5)
#AFFICHAGE 
#5. Affichage des resultats
plt.subplot(2,2,1)
plt.title("Gaussian")
plt.bar(np.array(["1","2","3","4","5"]),score_gaussian, align = "center", alpha = 0.5)

plt.subplot(2,2,2)
plt.bar(np.array(["1","2","3","4","5"]),score_Nearest, align = "center", alpha = 0.5)
plt.title("Nearest Centroid")

plt.subplot(2,2,3)
plt.bar(np.array(["1","2","3","4","5"]),score_kneighbors, align = "center", alpha = 0.5)
plt.title("K Neighbors")

plt.subplot(2,2,3)
plt.bar(np.array(["1","2","3","4","5"]),score_lmlr, align = "center", alpha = 0.5)
plt.title("Regression logistique")
plt.show()

y = np.array([np.mean(score_gaussian), np.mean(score_Nearest), np.mean(score_kneighbors) , np.mean(score_lmlr)])
x = np.array(['gaussian', 'NearestCentroid', 'kneighbors' , 'Logistic Regression'])

plt.title("Diagramme en barres ")
plt.bar(x, y, align = "center", alpha = 0.5 )
plt.show()
