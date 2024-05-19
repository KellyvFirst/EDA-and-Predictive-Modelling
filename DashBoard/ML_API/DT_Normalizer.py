#Import Librairies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
#st.set_option('deprecation.showPyplotGlobalUse', False)
#from FLASK.Client_Streamlit.eda import eda

#Load Data 
dataset = load_iris()

#Create dataframe with iris data
#print(dataset) 
#Recup input
data_input = dataset.data
target_names = dataset.target_names #CLasses
feature_names = dataset.feature_names # Columns
target = dataset.target #Outputs
#Creer dataFrame 
df = pd.DataFrame(data_input , columns = feature_names)
# Make target a series pour avoir accès aux valeurs 
target = pd.Series(target)

#Split data
X = df.values
Y = target.values

#Training
test_proportion = 0.30
seed = 5
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_proportion, random_state=seed)

#Data Transform
scaler = Normalizer()

#Pour faire le train et le test, il faut transformer les deux 
#On commence par tranfromer le Train et on utilise le parametre utilisé afin de transformer le Test après!
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) #X_test  = scaler.fit_transform(X_test)

#Define and train model / Model de classification 
classifier = DecisionTreeClassifier()
classifier.fit(X_train,Y_train)

#Evaluer la performence 
y_pred = classifier.predict(X_test)
Y_prob = classifier.predict_proba(X_test)

print('Accuracy : ', accuracy_score(Y_test, y_pred)) 
print('Recall : ', recall_score(Y_test, y_pred, average='macro')) 
print('F1 Score : ', f1_score(Y_test, y_pred, average = "weighted")) 
print('Precision : ', precision_score(Y_test, y_pred, average='micro'))
print('Matrix Confusion : ', confusion_matrix(Y_test, y_pred)) 

#Sauvegarder le model en memoire + Transformation
import pickle
#Save model
model_file = "classifer_DT_Norm.pickle"
pickle.dump(classifier, open(model_file, "wb")) #wb pour écrire / write #dump, il prend le model et il le copie dans notre fichier "classifer.pickle"
#Save transform
scaler_file = "sc_DT_Norm.pickle"
pickle.dump(scaler, open(scaler_file, "wb"))











