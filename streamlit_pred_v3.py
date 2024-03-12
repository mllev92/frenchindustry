import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pickle

df=pd.read_csv("df_full_full_catpos.csv")

st.title("French Industry")
st.sidebar.title("Sommaire")
pages=["Model1", "Model2","Model3"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[1] :
    st.write('### Prédiction salaire selon les caractéristiques de la personne (ML)')
    df=pd.read_csv("df_full_full_catpos.csv")
    df.drop_duplicates(inplace=True)
    df_debase = df
    df_debase = df_debase.drop(['nom_DEP','nom_REG','salaire_moyen'],axis=1)
    df=df.drop(['CODGEO', 'LIBGEO',"nom_REG", 'nom_DEP'],axis=1)

    X = df.drop('salaire_moyen', axis=1) 
    y = df['salaire_moyen']

    def charger_modele():
        # Charger le modèle à partir du fichier Pickle
        with open('modele.pkl', 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
        return modele

    X=df.drop('salaire_moyen', axis=1) 

    #DONNEES DE LA PERSONNE
    liste_ville = df_debase["LIBGEO"].drop_duplicates().tolist()

    commune = st.selectbox('Ville', liste_ville)

    genre = st.selectbox('Sexe',["H","F"])
    cat_poste = st.selectbox('Type Poste',["cadre sup", "cadre","travailleur","ouvrier"])
    ma_ligne = df_debase.loc[df_debase["LIBGEO"]==commune].iloc[[0]]

    if genre == "H" :
        ma_ligne["genre"]=1
    if genre == "F" :
        ma_ligne["genre"]=2

    if cat_poste == "cadre sup":
        ma_ligne["cat_poste"]=4
    if cat_poste == "cadre":
        ma_ligne["cat_poste"]=3
    if cat_poste == "travailleur":
        ma_ligne["cat_poste"]=2
    if cat_poste == "ouvrier":
        ma_ligne["cat_poste"]=1
    
    ma_ligne=ma_ligne.drop(["LIBGEO","CODGEO"],axis=1)    
    X = pd.concat([X, ma_ligne], ignore_index=True)

    scaler = StandardScaler()
    X_scaled_predict = scaler.fit_transform(X)
    ma_ligne = X_scaled_predict[-1]

    #JOUER LA PREDICTION
    caracteristiques = [ma_ligne]

    # Prévoir la classe avec le modèle
    modele = charger_modele()
    prediction = modele.predict(caracteristiques)
    st.write("le salaire horaire est",round(prediction[0],2))

if page == pages[0]:
    st.write('### Prédiction salaire selon les caractéristiques de la ville (ML)')

    df = pd.read_csv('df_full_full_propcadre.csv')
    df.drop_duplicates(inplace=True)

    df.rename(columns = {'LIBGEO' : 'nom_commune'}, inplace = True)
    dfcad = pd.read_csv('prop_cadre.csv')
    dfcad.drop_duplicates(subset=["nom_commune"], keep= 'first', inplace= True)
    dfcad1 = df.merge(dfcad, on = 'nom_commune', how = 'left')
    dfter = pd.read_csv('prop_tertiaire.csv')
    dfter.rename(columns = {'LIBGEO' : 'nom_commune'}, inplace = True)
    dfter.drop_duplicates(subset=["nom_commune"], keep= 'first', inplace= True)
    dftot = dfcad1.merge(dfter, on = 'nom_commune', how = 'left')
    dftot.dropna(inplace = True)
    df = dftot
    df.DEP = df.DEP.replace("2A","222")
    df.DEP = df.DEP.replace("2B","223")
    df_debase = df
    df = df.drop(['nom_commune',"CODGEO",'nom_DEP','nom_REG','nom_commune',"Departement"],axis=1)
    df_debase = df_debase.drop(['nom_DEP','nom_REG',"Departement",'salaire_moyen'],axis=1)
    X = df.drop('salaire_moyen', axis=1) 
    y = df['salaire_moyen']

    def charger_modele():
    # Charger le modèle à partir du fichier Pickle
        with open('modele2.pkl', 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
        return modele

    liste_ville = df_debase["nom_commune"].drop_duplicates().tolist()

    commune = st.selectbox('Ville', liste_ville)
    genre = st.selectbox('Sexe',["H","F"])
    age = st.number_input('Votre âge', min_value= 18 ,step = 1)
    cadre = st.selectbox("Etes-vous cadre",["Oui","Non"])
    ma_ligne = df_debase.loc[df_debase["nom_commune"]==commune]

    #traitement du genre
    if genre != "":
        if genre == "F" :
            ma_ligne["ratio_homme"] = 0
            ma_ligne["ratio_femme"] = 1
        elif genre =="H":
            ma_ligne["ratio_femme"] = 0
            ma_ligne["ratio_homme"] = 1
        
    ma_ligne = df_debase.loc[df_debase["nom_commune"]==commune]
 
    ma_ligne["age_moyen"]=age

    #if cadre == True :
    #    ma_ligne["proportion de cadres"]=0.145967
    #if cadre == False :
    #    ma_ligne["proportion de cadres"]=0
        
    ma_ligne=ma_ligne.drop(["nom_commune","CODGEO"],axis=1)    
    X = pd.concat([X, ma_ligne], ignore_index=True)

    scaler = StandardScaler()
    X_scaled_predict = scaler.fit_transform(X)
    ma_ligne = X_scaled_predict[-1]

    #JOUER LA PREDICTION
    caracteristiques = [ma_ligne]

    # Prévoir la classe avec le modèle
    modele = charger_modele()
    prediction = modele.predict(caracteristiques)
    st.write("le salaire horaire est",round(prediction[0],2))
    
    
if page == pages[2]:
    st.write('### Prédiction selon les caractéristiques de la personne via moyenne')

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    dfsal = pd.read_csv("net_salary_per_town_categories.csv")

    def devine_salaire(genre, ville, age, profession,dep="") :
        dfsal = pd.read_csv("net_salary_per_town_categories.csv")
        dfsal= dfsal.rename(columns={'SNHMFC14': 'Salaire_cadre_F',
                          "SNHMFP14":"Salaire_cadremoy_F",
                          "SNHMFE14":"Salaire_employe_F",
                          "SNHMFO14":"Salaire_travailleur_F",
                          "SNHMHC14":'Salaire_cadre_H',
                          "SNHMHP14":"Salaire_cadremoy_H",
                          "SNHMHE14":"Salaire_employe_H",
                          "SNHMHO14":"Salaire_travailleur_H",
                          "SNHMF1814":"Salaire_F_18-25",
                          "SNHMF2614":"Salaire_F_26-50",
                          "SNHMF5014":"Salaire_F_50",
                          "SNHMH1814":"Salaire_H_18-25",
                          "SNHMH2614":"Salaire_H_26-50",
                          "SNHMH5014":"Salaire_H_50",
                        "SNHM1814":"Salaire_ALL_18-25",
                                 "SNHM2614":"Salaire_ALL_26-50",
                                 "SNHM5014":"Sallaire_ALL_50",
                                 "SNHMF14":"Salaire_F",
                                 "SNHMH14":"Salaire_H",
                                 "SNHM14":"Salaire_ALL",
                                 "SNHMC14":"Salaire_ALL_cadre",
                                 "SNHMP14":"Salaire_ALL_cadremoy",
                                 "SNHME14":"Salaire_ALL_employe",
                                 "SNHMO14":"Salaire_ALL_travailleur"
                         })
        doublon = dfsal[dfsal["LIBGEO"].duplicated()][["LIBGEO","CODGEO"]]
        doublon["dep"]=doublon.CODGEO.str.slice(0, 2)
        dfsal["dep"]=dfsal.CODGEO.str.slice(0, 2)

        ##cadre = 4
        ##cadre moyen = 3
        ##employe = 2
        ##travailleur = 1
        if ville in doublon.LIBGEO.values and dep=="":
            raise ValueError("Ville en doublon, précisez le numéro de département")
        elif ville in doublon.LIBGEO.values :
            CODGEO = dfsal[(dfsal.LIBGEO == ville)&(dfsal.dep == dep)].CODGEO.values[0]
            #print(CODGEO)
            ma_ville = dfsal[dfsal.CODGEO == CODGEO]
        else:
            ma_ville = dfsal[dfsal.LIBGEO == ville]

        #traitement des professions
        if profession == 4 :
            prof = "cadre"
        elif profession ==3 :
            prof = "cadremoy"
        elif profession ==2:
            prof = "employe"
        elif profession == 1 :
            prof = "travailleur"

        #traitement des ages
        if age <26 :
            groupe = "18-25"
        elif age <51:
            groupe = "26-50"
        elif age > 50 :
            groupe = "50"

       #création des noms de colonnes
        col1 = "Salaire_"+prof+"_"+genre
        col2 = "Salaire_"+genre+"_"+groupe

        ma_ville = ma_ville[[col1,col2]]

        #print(ma_ville)
        return(round((2*ma_ville.iloc[0,0]+ma_ville.iloc[0,1])/3))
    
    liste_ville = dfsal["LIBGEO"].drop_duplicates().tolist()
    commune = st.selectbox('Ville', liste_ville)
    genre = st.selectbox('Sexe',["H","F"])
    range_age = [n for n in range(18,100)]
    age = st.selectbox("Age", range_age)
    cat_poste = st.selectbox('Type Poste',["cadre sup", "cadre","travailleur","ouvrier"])

    if cat_poste == "cadre sup":
        cat_poste=4
    if cat_poste == "cadre":
        cat_poste=3
    if cat_poste == "travailleur":
        cat_poste=2
    if cat_poste == "ouvrier":
        cat_poste=1
    
    #prédiction
    prediction_moyenne=round(devine_salaire(genre, commune, age, cat_poste),2)
    st.write("le salaire horaire est",prediction_moyenne)
