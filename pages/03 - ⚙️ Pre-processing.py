import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import requests
from io import BytesIO


dfent = pd.read_csv("base_etablissement_par_tranche_effectif.csv", sep = ",") 
dfgeo = pd.read_csv ("name_geographic_information.csv", sep = ",")
dfsal = pd.read_csv("net_salary_per_town_categories.csv", sep = ",")
dfpop = pd.read_csv("population.csv", sep = ",",low_memory=False)

df_full = pd.read_csv("df_full.csv")
df_full_ratio_commune = pd.read_csv("df_full_ratio_commune.csv")
df_full_ratio_dept = pd.read_csv("df_full_ratio_dept.csv")

st.sidebar.markdown(
    """
    - **Cursus** : Data Analyst
    - **Formation** : Bootcamp
    - **Mois** : Janvier 2024
    - **Groupe** : 
        - Manon PEQUILLAT
        - Margaux POLOMACK
        - Yannick LIN""")

# Page d'exploration des données
if 1 == 1 :
    st.header("🔍 Exploration des Données")
    st.markdown(""" """ """<style>h1 {color: #4629dd;  font-size: 70px;/* Changez la couleur du titre h1 ici */} h2 {color: #440154ff;    font-size: 50px /* Changez la couleur du titre h2 ici */} h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown(""" """ """<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    if st.button("Entreprises") :
        #Afficher le Dataset dfent
        st.subheader("Dataset sur la répartition des entreprises en fonction des communes françaises")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(dfent.head())
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", dfent.shape[0])
        st.write("- Nombre de colonnes:", dfent.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(dfent.describe())
        st.write("")
        col1, col2 = st.columns([2,3])
        with col1 : 
            st.write("- Valeurs manquantes :", dfent.isnull().sum())
        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            dfent.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)
      
            st.write("")

    if st.button("Nom géographique") :
        #Afficher le Dataset dfgeo
        st.subheader("Dataset sur les informations géographiques des communes françaises")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(dfgeo.head())
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", dfgeo.shape[0])
        st.write("- Nombre de colonnes:", dfgeo.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(dfgeo.describe())
        st.write("")
        col1, col2 = st.columns([2,3])
        with col1 : 
            st.write("- Valeurs manquantes :", dfgeo.isnull().sum())
        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            dfgeo.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)

        st.write("")

    if st.button("Salaires") :
        #Afficher le Dataset dfsal
        st.subheader("Dataset sur la répartition des salaires moyens en fonction des communes françaises et différentes caractéristiques")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(dfsal.head())
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", dfsal.shape[0])
        st.write("- Nombre de colonnes:", dfsal.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(dfsal.describe())
        st.write("")
        col1, col2 = st.columns([2,3])
        with col1 : 
            st.write("- Valeurs manquantes :", dfsal.isnull().sum())
        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            dfsal.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)

        st.write("")

    if st.button("Population") :
        #Afficher le Dataset dfent
        st.subheader("Dataset sur les caractéristiques de la population en fonction des communes françaises")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(dfpop.head())
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", dfpop.shape[0])
        st.write("- Nombre de colonnes:", dfpop.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(dfpop.describe())
        st.write("")
        col1, col2 = st.columns([2,3])
        with col1 : 
            st.write("- Valeurs manquantes :", dfpop.isnull().sum())
        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            dfpop.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)

        st.write("")
