# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler 

def eda(df, target_names, feature_names, target) :
    #st.write('You selected EDA')
    st.subheader("Exploratory Data Analysis and Visualization")
    transform_options = [
                            "None", 
                            "StandardScaler", 
                            "Normalizer", 
                            "MinMaxScaler"]
                
    transform = st.selectbox("Select data transform",
                                            transform_options)
    st.write("Choose a plot type from the options below :")

    #SCALERS AND NEW DATAFRAMES 
    #StandardScaler
    scaler_stand = StandardScaler()
    stand = scaler_stand.fit_transform(df)
    df_stand = pd.DataFrame(stand,columns = feature_names)
    #Normalizer
    scaler_norm = Normalizer()
    norm = scaler_norm.fit_transform(df)
    df_norm = pd.DataFrame(norm,columns = feature_names)
    #MinMaxScaler
    scaler_min = MinMaxScaler()
    min = scaler_min.fit_transform(df)
    df_min = pd.DataFrame(min,columns = feature_names)

     #Add option to Show missing value
    if st.checkbox("Show missing value"):
        st.write(df.isna().sum())
    if st.checkbox("Show data type"):
        st.write(df.dtypes)

    #Add option to show/hide data
    if st.checkbox("Show raw data"):
        if transform == 'None' :
            st.markdown("### **Données Originales :**")
            st.write(df)
        elif transform == 'StandardScaler' :
            st.markdown("### **Données avec transform StandardScaler :**")
            st.write(stand)
        elif transform == 'Normalizer' :
            st.markdown('### **Données avec transform Normalizer :**')
            st.write(norm)
        elif transform == 'MinMaxScaler' :
            st.markdown('### **Données avec transform MinMaxScaler :**')
            st.write(min)
        
    if st.checkbox("Show descriptive Statistics"):
        
        if transform == 'None' :
            st.markdown("### **Données Originales :**")
            st.write(df.describe())
        elif transform == 'StandardScaler' :
            st.markdown("### **Données avec transform StandardScaler :**")
            st.write(df_stand.describe())
        elif transform == 'Normalizer' :
            st.markdown('### **Données avec transform Normalizer :**')
            st.write(df_norm.describe())
        elif transform == 'MinMaxScaler' :
            st.markdown('### **Données avec transform MinMaxScaler :**')
            st.write(df_min.describe())

    if st.checkbox("Show correlation matrix"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if transform == 'None' :
            st.markdown("### **Données Originales :**")
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm")
            st.pyplot()
        elif transform == 'StandardScaler' :
            st.markdown("### **Données avec transform StandardScaler :**")
            corr = df_stand.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm")
            st.pyplot()
        elif transform == 'Normalizer' :
            st.markdown('### **Données avec transform Normalizer :**')
            corr = df_norm.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm")
            st.pyplot()
        elif transform == 'MinMaxScaler' :
            st.markdown('### **Données avec transform MinMaxScaler :**')
            corr = df_min.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm")
            st.pyplot()

    if st.checkbox("Show Histogram for each attributes"):
        if transform == 'None' :
            st.markdown("### **Données Originales :**")
            for col in df.columns:
                fig, ax = plt.subplots()
                ax.hist(df[col], bins=20, density=True, alpha=0.5)
                ax.set_title(col)
                st.pyplot(fig)
        elif transform == 'StandardScaler' :
            st.markdown("### **Données avec transform StandardScaler :**")
            for col in df_stand.columns:
                fig, ax = plt.subplots()
                ax.hist(df_stand[col], bins=20, density=True, alpha=0.5)
                ax.set_title(col)
                st.pyplot(fig)
        elif transform == 'Normalizer' :
            st.markdown('### **Données avec transform Normalizer :**')
            for col in df_norm.columns:
                fig, ax = plt.subplots()
                ax.hist(df_norm[col], bins=20, density=True, alpha=0.5)
                ax.set_title(col)
                st.pyplot(fig)
        elif transform == 'MinMaxScaler' :
            st.markdown('### **Données avec transform MinMaxScaler :**')
            for col in df_norm.columns:
                fig, ax = plt.subplots()
                ax.hist(df_norm[col], bins=20, density=True, alpha=0.5)
                ax.set_title(col)
                st.pyplot(fig)

    if st.checkbox("Show Density for each attributes"):
        if transform == 'None' :
            st.markdown("### **Données Originales :**")
            for col in df.columns:
                fig, ax = plt.subplots()
                sns.kdeplot(df[col], fill=True)
                ax.set_title(col)
                st.pyplot(fig)
        elif transform == 'StandardScaler' :
            st.markdown("### **Données avec transform StandardScaler :**")
            for col in df_stand.columns:
                fig, ax = plt.subplots()
                sns.kdeplot(df_stand[col], fill=True)
                ax.set_title(col)
                st.pyplot(fig)
        elif transform == 'Normalizer' :
            st.markdown('### **Données avec transform Normalizer :**')
            for col in df_norm.columns:
                fig, ax = plt.subplots()
                sns.kdeplot(df_norm[col], fill=True)
                ax.set_title(col)
                st.pyplot(fig)
        elif transform == 'MinMaxScaler' :
            st.markdown('### **Données avec transform MinMaxScaler :**')
            for col in df_min.columns:
                fig, ax = plt.subplots()
                sns.kdeplot(df_min[col], fill=True)
                ax.set_title(col)
                st.pyplot(fig)

    if st.checkbox("Show Scatter plot"):
        if transform == 'None' :
            st.markdown("### **Données Originales :**")
            fig = px.scatter(df, x= feature_names[0], y = feature_names[1], color=target)
            st.plotly_chart(fig) 
        elif transform == 'StandardScaler' :
            st.markdown("### **Données avec transform StandardScaler :**")
            fig = px.scatter(df_stand, x= feature_names[0], y = feature_names[1], color=target)
            st.plotly_chart(fig)
        elif transform == 'Normalizer' :
            st.markdown('### **Données avec transform Normalizer :**')
            fig = px.scatter(df_norm, x= feature_names[0], y = feature_names[1], color=target)
            st.plotly_chart(fig)
        elif transform == 'MinMaxScaler' :
            st.markdown('### **Données avec transform MinMaxScaler :**')
            fig = px.scatter(df_min, x= feature_names[0], y = feature_names[1], color=target)
            st.plotly_chart(fig)

