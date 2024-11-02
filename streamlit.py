import streamlit as st

st.sidebar.title('Escolha os Atributos')

st.sidebar.write("Selecione os atributos para classificação:")

atributos = ["Age", "Sex", "Pclass", "Fare"]



atributos_selecionados = st.sidebar.multiselect("Atributos", options=atributos)
if st.sidebar.button("Treinar Modelo"): 
    st.sidebar.write("Treinando o modelo com os atributos:", atributos_selecionados)