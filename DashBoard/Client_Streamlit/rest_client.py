import streamlit as st
import requests
import json
from eda import eda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

url="http://localhost:8004/train"

#Load Data 
dataset = load_iris()

#Create dataframe with iris data
#print(dataset) 
#Recup input
data = dataset.data
target_names = dataset.target_names #CLasses
feature_names = dataset.feature_names # Columns
target = dataset.target #Outputs
#Creer dataFrame 
df = pd.DataFrame(data , columns = feature_names)
# Make target a series pour avoir accès aux valeurs 
target = pd.Series(target)

#Streamlit
#Set up App 

st.set_page_config(page_title="EDA and ML Dashboard",
                   layout="centered",
                   initial_sidebar_state="auto")

st.title("EDA and Predictive Modelling Dashboard")

options = ["EDA", "Predictive Modelling"]
selected_option = st.sidebar.selectbox("Select an option", options)
if selected_option == "EDA":
    # Call/invoke EDA function from ead.py
    eda(df, target_names, feature_names, target)
elif selected_option == "Predictive Modelling":
    st.subheader("Predictive Modelling")
    st.write("Choose a transform type and Model from the options below:")

    #User Input / Recup des données client 
    def user_input_choice():
        transform_options = ["None",
                         "StandardScaler",
                         "Normalizer",
                         "MinMaxScaler"]
        transform = st.selectbox("Select data transform",
                             transform_options)

        classifier_list = ["LogisticRegression",
                       "SVM",
                       "DecisionTree",
                       "KNeighbors",
                       "RandomForest"]
        classifier = st.selectbox("Select classifier", classifier_list)
        input_data = {
                'transfom' : transform, 
                'model' : classifier
            }
    
        features = pd.DataFrame(input_data, index=[0]) # Create a dataframe with input values
        return features
    
    df = user_input_choice() 
    array = df.values
    # transform = array[0,0]
    # st.write(transform) 
    # classifier = array[0,1]
    # st.write(classifier)
    transform, classifier = array.ravel()

    #Create json object with transform and model
    request_data = json.dumps({'transform' : transform, 'classifier' : classifier}) 

    #Send to server / Endpoint (REST_API)
    requestpost = requests.post(url, request_data)
    # st.write(requestpost.json()['accuracy'])
    #st.write(requestpost)

    #Recup DATA from REST_API 
    #Recup de la réponse sous forme d'objet JSON
    res = requestpost.json()
    #st.write(res)
    accuracy = res['accuracy']
    recall = res['recall']
    f1 = res['f1_score']
    precision = res['precision']
    confusion = json.loads(res['matrix_c']) #afficher sous forme de liste 
    row = 3
    colm = 3
    matrix = np.reshape(confusion, (row, colm))
   
    st.write('Your choice is :' , transform , 'et' , classifier)
    st.write("Accuracy : " , accuracy)
    st.write("Recall : ", recall)
    st.write("F1_score : " , f1)
    st.write("Precision : " , precision)
    st.write("Confusion Matrix: " , matrix)

   