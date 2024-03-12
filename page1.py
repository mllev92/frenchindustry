import streamlit as st
st.title("💶 French Industry 💶")
st.sidebar.title("Sommaire")
pages=["👋 Introduction", "📊 DataViz","⚙️ Pre-processing",'🧠 Modèles de Machine Learning',"🔮 Prédictions"]
page=st.sidebar.radio("⬇️ Aller vers", pages)
