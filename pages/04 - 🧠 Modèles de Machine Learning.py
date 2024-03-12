import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import statsmodels.api

dfent = pd.read_csv("base_etablissement_par_tranche_effectif.csv")
dfgeo = pd.read_csv ("name_geographic_information.csv")
dfsal = pd.read_csv("net_salary_per_town_categories.csv")
#dfpop = pd.read_csv("population.csv",low_memory=False)

df1 = pd.read_csv("df_ratio.csv", index_col="CODGEO")
df2 = pd.read_csv("df_full_cadre_tertiaire.csv", index_col="CODGEO")
df3 = pd.read_csv("df_full_cadre_tertiaire_tag_dep.csv", index_col="CODGEO")
df4 = pd.read_csv("df_cat_poste.csv", index_col="CODGEO")

if 1 == 1 :
    st.title("💶 French Industry 💶")
    
    st.markdown("# 🧠 Modèles de Machine Learning")
    st.sidebar.markdown("# 🧠 Modèles de Machine Learning")

    
    st.subheader("Objectif")
    st.write("Prédire le salaire moyen d'une personne (variable continue) en fonction des données contenues dans les variables explicatives qui le composent.")
    
    st.write("")
    st.write("")

    if st.button("Modèles de régression") :
        st.subheader("Choix des modèles")
        st.markdown("""
                    Afin de prédire le salaire moyen, nous avons étudié la performance de plusieurs modèles de machine learning :
                    - Régression linéaire
                    - Ridge
                    - Lasso
                    - Ridge
                    - Elastic Net
                    - KNN
                    - Decision Tree
                    - Random Forest
                    - SVR.

                    Ces modèles ont été appliqués à 4 Dataframes issus de pré-processing différent afin de comparer les résultats et de déterminer le meilleur modèle.
        """)

        st.write("")
        st.write("")

    # Présentation du jeu de modélisation n°1
        
    if st.button("Choix du jeu de modélisation"):
        st.subheader("Jeu de modélisation n°1")
        st.markdown("""
                    Ce DataFrame a été créé à partir des données fournies pour le projet et avec les variables (ensemble de ratios sur la population et les entreprises) créées lors de l’initialisation des matrices de corrélation
                    """)
        st.write("")
        st.write("")
        st.dataframe(df1.head())

        st.markdown("#### Tableau global des résultats")
        result1 = pd.read_csv("result_model1.csv", index_col = 1)

        result1.drop(columns=result1.columns[0], axis=1, inplace=True)

        st.dataframe(result1)

        st.markdown("""
                    #### Interprétation


                    On constate des R² plutôt faibles (autour de 0.4, voire 0.2) ou du sur apprentissage lorsque le R² est meilleur (pour Random Forest et SVR).  
                    
                    
                    La MSE sur l’ensemble de test est assez élevée. La RMSE de test, à son minimum est à 1.42 (donc un écart moyen de 1.42€ sur le salaire moyen horaire prédit).  
                    
                    
                    Pour ce premier jeu de modélisation, on peine donc à trouver un “meilleur” modèle. Nous souhaitons donc améliorer notre jeu de données.  
                    """)
        st.write("")
        st.write("")


    # Présentation du jeu de modélisation n°2
        
        st.subheader("Jeu de modélisation n°2")
        st.markdown("""
                    Sur la base du DataFrame précédent, nous avons ajouté :
                    - La proportion de cadre par commune en 2020
                    - La proportion de personnes travaillant dans le tertiaire par commune en 2020

                    """)
        st.write("")
        st.write("")
        st.dataframe(df2.head())

        st.markdown("#### Tableau global des résultats")
        # Insertion du dataframe résultats

        result2 = pd.read_csv("result_model2.csv", index_col = 1)

        result2.drop(columns=result2.columns[0], axis=1, inplace=True)
        st.dataframe(result2)

        st.markdown("""
                    #### Interprétation
                    Les modèles KNN et SVR semblent les plus intéressants (les écarts entre R² train et R² test sont les plus faibles, tout en étant proche de 1). Les écarts de prédictions (RMSE respectivement à 1.28 et 1.11).
                    
                    
                    Le Random Forest a une RMSE faible mais l’écart entre les valeurs de R² témoignent d’un surapprentissage.
                    
                    
                    Afin d'affiner ces résultats, nous effectuons un GridSearch pour évaluer les meilleurs paramètres à passer pour chacun de ces modèles.

                    """)
        st.write("")
        st.write("")

        st.markdown("#### Résultats après Gridsearch")
        # Insertion résultat gridsearch
        data = {
        'Model': ['KNN', 'SRV'],
        'R2_train': [0.859765, 0.885198],
        'R2_test': [0.681018, 0.802175],
        'R2_full': [0.815537,0.864396],
        'MAE_train': [0.628312, 0.772503],
        'MAE_test': [0.934093, 1.180191],
        'MSE_train': [0.912875, 0.476711],
        'MSE_test': [2.114212, 0.663887],
        'RMSE_train': [0.955445, 0.878921],
        'RMSE_test': [1.454033, 1.086366]
                }
        
        tab = pd.DataFrame(data)
        tab.set_index('Model', inplace = True)
        st.dataframe(tab)

        st.markdown("""
                    Le modèle KNN a un surapprentissage beaucoup trop important. Le modèle SVR a une meilleure erreur quadratique, sans pour autant perdre la qualité de son écart entre r2 train et test.

                    
                    On constate en observant l’importance des variables explicatives dans les différents modèles que la proportion de cadres est désormais la variable la plus importante.

                    
                    Est-il possible d’améliorer nos scores avec l’ajout d’une donnée sur le département d’appartenance (comme pour l’appartenance à la région Ile-de-France) ?


                    """)
        st.write("")
        st.write("")


    # Présentation du jeu de modélisation n°3

        st.subheader("Jeu de modélisation n°3")
        st.markdown("""
                    Sur base du DataFrame précédent, nous avons ajouté une colonne par département encodé selon si la commune fait partie (1) ou non (0) à ce département (OneHotEncoder).
                            
                            """)

        st.write("")
        st.write("")
        st.dataframe(df3.head())

        st.markdown("#### Tableau global des résultats")
        # Insertion du dataframe résultats
        result3 = pd.read_csv("result_model3.csv", index_col = 1)

        result3.drop(columns=result3.columns[0], axis=1, inplace=True)

        st.dataframe(result3)

        st.markdown("""
                    #### Interprétation
                    Les résultats ne sont pas sensiblement meilleurs. Le SVR est un plus grand surapprentissage que le jeu de données précédent, sans pour autant améliorer l’erreur quadratique.
                    """)
        st.write("")
        st.write("")


    # Présentation du jeu de modélisation n°4
        
        st.subheader("Jeu de modélisation n°4")
        st.markdown("""
                    Ce 4e DataFrame a été créé depuis le fichier net_salary.csv fourni pour le projet. Les colonnes ont été pivoté afin de catégoriser le salaire moyen selon le genre et la catégorie de poste :
                    - genre = 1 : femme / 2 : homme
                    - cat_poste = 1 : travailleur(euse) / 2 : employé(e) / 3 : cadre moyen / 4 : cadre

                    """)
        st.write("")
        st.write("")
        st.dataframe(df4.head())

        st.markdown("#### Tableau global des résultats")
        # Insertion du dataframe résultats
        # result3 = pd.read_csv("result_model4.csv", index_col = 1)

        # result3.drop(columns=result3.columns[0], axis=1, inplace=True)

        # st.dataframe(result4)

        st.markdown("""
                    #### Interprétation
                    On note un R² de test supérieur au R² train sur le modèle SVR. Cependant, les valeurs des RMSE de test sont moins intéressantes que sur le DataFrame n°2.

                    
                    Il est intéressant de noter que pour SVR, nous avons un DataFrame de grandeur optimale d’après la courbe d’apprentissage. 

                    
                    Le modèle SVR offre les meilleurs résultats. Si on analyse de plus près le graphique de dispersions des résidus de ce modèle, on constate de nombreux résidus avec une valeur parfois supérieure à 5, ce qui est très important pour un salaire horaire.

                    """)
        st.write("")
        st.write("")
    
        st.subheader("Modèle le plus performant")

        st.markdown("""
                    En conclusion, le jeu de données avec la proportion de cadre semble offrir le meilleur compromis en termes de performance (jeu de modélisation n°2 avec le modèle SVR).

                     
                    Les modèles Random Forest obtiennent de bons résultats d’erreur quadratique, malheureusement, l’écart des R² est trop important signe de surapprentissage.

                    """)
        st.write("")
        st.write("")

    if st.button("Evaluation graphique du modèle") :
        
        #Courbe d'apprentissage
        st.subheader("Courbe d'apprentissage du modèle")
        # insérer la courbe d'apprentissage

        image_apprentissage = "https://zupimages.net/up/24/11/m3js.png"

        st.image(image_apprentissage, use_column_width=True)

        st.markdown("""
                    ##### Points à retenir :         
                    - Modèle qui s'ajuste au fur à mesure aux données d'entrainement.
                    - Pas de soupçon d'overfitting
                    """)
        
        st.write("")
        st.write("")

        #QQplot, residu et prediction vs vraies
        st.subheader("Graphique des résidus et QQ-plot")
        # insérer qqplot et résidu
        image_qqplot = "https://zupimages.net/up/24/11/n6me.png"

        st.image(image_qqplot, use_column_width=True)

        st.write("")
        st.write("")

        st.markdown("""
                    ##### Points à retenir :      
                    - Distribution relativement centrée autour de 0 et entre -2.5 et 2.5
                    - Quelques valeurs extrêmes.

                    """)
        
        st.write("")
        st.write("")


    if st.button("Features importance") :
        
        st.markdown("""
                    Le SRV n'ayant pas de Features importance, nous allons présenter ceux du RandomForestRegressor, un modèle qui donne aussi des résultats très satisfaisant.
                    """
                    )

        # Visualiser les importances des caractéristiques
        st.subheader("Importance des variables du RandomForestRegressor")
        
        # insérer le graphe
        image_importances = "https://zupimages.net/up/24/11/82nd.png"

        st.image(image_importances, use_column_width=True)
        st.markdown("""
                    On remarque que la variable la plus importante est le ratio_cadre qui représente 70% pour le modèle.
                    """)
