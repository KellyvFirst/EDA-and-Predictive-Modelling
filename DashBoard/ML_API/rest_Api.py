#Import Librairies
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import pickle
import json 
#dataset
dataset = load_iris()
data = dataset.data
target_names = dataset.target_names
feature_names = dataset.feature_names
target = dataset.target
df = pd.DataFrame(data, columns=feature_names)
print('Dataframe:', df)
# Afficher les noms des caractéristiques (feature_names)
print('Feature names:', dataset.feature_names)
# Afficher les noms des cibles (target_names)
print('Target names:', dataset.target_names)
# Afficher la cible (target) pour le premier élément du jeu de données
print('Target for the first instance:', dataset.target[0])

# Afficher les caractéristiques (features) pour le premier élément du jeu de données
print('Features for the first instance:', df.iloc[0])
target = pd.Series(target)
print('Targets are', target);
app = Flask(__name__)


# Load Models
#DecisionTreeClassifier
classifer_DT_MinMax = pickle.load(open('classifer_DT_MinMax.pickle','rb'))
classifer_DT_Norm = pickle.load(open('classifer_DT_Norm.pickle','rb'))
classifer_DT_Origi = pickle.load(open('classifer_DT_Origi.pickle','rb'))
classifer_DT_Stand = pickle.load(open('classifer_DT_Stand.pickle','rb'))
#KNeighborsClassifier
classifer_KN_MinMax = pickle.load(open('classifer_KN_MinMax.pickle','rb'))
classifer_KN_Norm = pickle.load(open('classifer_KN_Norm.pickle','rb'))
classifer_KN_Origi = pickle.load(open('classifer_KN_Origi.pickle','rb'))
classifer_KN_Stand = pickle.load(open('classifer_KN_Stand.pickle','rb'))
#LogisticRegression 
classifer_LR_MinMax = pickle.load(open('classifer_LR_MinMax.pickle','rb'))
classifer_LR_Norm = pickle.load(open('classifer_LR_Norm.pickle','rb'))
classifer_LR_Origi = pickle.load(open('classifer_LR_Origi.pickle','rb'))
classifer_LR_Stand = pickle.load(open('classifer_LR_Stand.pickle','rb'))
#RandomForestClassifier
classifer_RF_MinMax= pickle.load(open('classifer_RF_MinMax.pickle','rb'))
classifer_RF_Norm = pickle.load(open('classifer_RF_Norm.pickle','rb'))
classifer_RF_Origi = pickle.load(open('classifer_RF_Origi.pickle','rb'))
classifer_RF_Stand = pickle.load(open('classifer_RF_Stand.pickle','rb'))
#SVM
classifer_SVM_MinMax = pickle.load(open('classifer_SVM_MinMax.pickle','rb'))
classifer_SVM_Norm = pickle.load(open('classifer_SVM_Norm.pickle','rb'))
classifer_SVM_Origi = pickle.load(open('classifer_SVM_Origi.pickle','rb'))
classifer_SVM_Stand = pickle.load(open('classifer_SVM_Stand.pickle','rb'))

# Load Transforms
#DecisionTreeClassifier
sc_DT_MinMax = pickle.load(open('sc_DT_MinMax.pickle','rb'))
sc_DT_Norm = pickle.load(open('sc_DT_Norm.pickle','rb'))
sc_DT_Stand = pickle.load(open('sc_DT_Stand.pickle','rb'))
#KNeighborsClassifier
sc_KN_MinMax = pickle.load(open('sc_KN_MinMax.pickle','rb'))
sc_KN_Norm = pickle.load(open('sc_KN_Norm.pickle','rb'))
sc_KN_Stand = pickle.load(open('sc_KN_Stand.pickle','rb'))
#LogisticRegression 
sc_LR_MinMax = pickle.load(open('sc_LR_MinMax.pickle','rb'))
sc_LR_Norm = pickle.load(open('sc_LR_Norm.pickle','rb'))
sc_LR_Stand = pickle.load(open('sc_LR_Stand.pickle','rb'))
#RandomForestClassifier
sc_RF_MinMax = pickle.load(open('sc_RF_MinMax.pickle','rb'))
sc_RF_Norm = pickle.load(open('sc_RF_Norm.pickle','rb'))
sc_RF_Stand = pickle.load(open('sc_RF_Stand.pickle','rb'))
#SVM
sc_SVM_MinMax = pickle.load(open('sc_SVM_MinMax.pickle','rb'))
sc_SVM_Norm = pickle.load(open('sc_SVM_Norm.pickle','rb'))
sc_SVM_Stand = pickle.load(open('sc_SVM_Stand.pickle','rb'))

@app.route('/train', methods=['POST'])
def train_model():
    # Parse input data
    input_data = request.get_json(force=True)
    transform = input_data['transform']
    model = input_data['classifier']

     # Split data
    X = df.values
    Y = target.values
    test_proportion = 0.30
    seed = 5
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_proportion, random_state=seed)

    if transform == 'None' and model == 'LogisticRegression':
        #Evaluer la performence 
        y_pred = classifer_LR_Origi.predict(X_test)
    elif transform == 'StandardScaler' and model == 'LogisticRegression':
        y_pred = classifer_LR_Stand.predict(X_test)
    elif transform == 'Normalizer' and model == 'LogisticRegression':
        y_pred = classifer_LR_Norm.predict(X_test)
    elif transform == 'MinMaxScaler' and model == 'LogisticRegression':
        y_pred = classifer_LR_MinMax.predict(X_test)
    elif transform == 'None' and model == 'SVM':
        y_pred = classifer_SVM_Origi.predict(X_test)
    elif transform == 'StandardScaler' and model == 'SVM':
        y_pred = classifer_SVM_Stand.predict(X_test)
    elif transform == 'Normalizer' and model == 'SVM':
        y_pred = classifer_SVM_Norm.predict(X_test)
    elif transform == 'MinMaxScaler' and model == 'SVM':
        y_pred = classifer_SVM_MinMax.predict(X_test)
    elif transform == 'None' and model == 'DecisionTree':
        y_pred = classifer_DT_Origi.predict(X_test)
    elif transform == 'StandardScaler' and model == 'DecisionTree':
        y_pred = classifer_DT_Stand.predict(X_test)    
    elif transform == 'Normalizer' and model == 'DecisionTree':
        y_pred = classifer_DT_Norm.predict(X_test)
    elif transform == 'MinMaxScaler' and model == 'DecisionTree':
        y_pred = classifer_DT_MinMax.predict(X_test)
    elif transform == 'None' and model == 'RandomForest':
        y_pred = classifer_RF_Origi.predict(X_test)
    elif transform == 'StandardScaler' and model == 'RandomForest':
        y_pred = classifer_RF_Stand.predict(X_test)   
    elif transform == 'Normalizer' and model == 'RandomForest':
        y_pred = classifer_RF_Norm.predict(X_test)
    elif transform == 'MinMaxScaler' and model == 'RandomForest':
        y_pred = classifer_RF_MinMax.predict(X_test)    
    elif transform == 'None' and model == 'KNeighbors':
        y_pred = classifer_KN_Origi.predict(X_test)
    elif transform == 'StandardScaler' and model == 'KNeighbors':
        y_pred = classifer_KN_Stand.predict(X_test)   
    elif transform == 'Normalizer' and model == 'KNeighbors':
        y_pred = classifer_KN_Norm.predict(X_test)
    elif transform == 'MinMaxScaler' and model == 'KNeighbors':
        y_pred = classifer_KN_MinMax.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred, average='macro') 
    f1 = f1_score(Y_test, y_pred, average = "weighted")
    precision = precision_score(Y_test, y_pred, average='micro')
    matrix_c = confusion_matrix(Y_test, y_pred)

    matr_arret = matrix_c.ravel() #Transformer en arrêt
    #print(matr_arret)
    matr_string = json.dumps(matr_arret.tolist()) #Transformer en liste 
    #print(matr_string)
    #return d'un objet JSON
    return jsonify({'accuracy' : accuracy, 
            'recall' : recall, 
            'f1_score' : f1,  
            'precision' : precision,
            'matrix_c' : matr_string})

   
if __name__ == "__main__":
    app.run(port=8004, debug=True)


