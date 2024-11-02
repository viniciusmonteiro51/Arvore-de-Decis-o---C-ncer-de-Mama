import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import streamlit as st

df = pd.read_csv('cancer_mama.csv')

df_data = df.drop(['diagnosis'], axis=1)
df_target = df['diagnosis'].map({'M': 1, 'B': 0})

x = df_data
y = df_target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

st.sidebar.title('Escolha os Atributos')
st.sidebar.write("Selecione os atributos para classificação:")

atributos = list(df_data.columns)

atributos_selecionados = st.sidebar.multiselect("Atributos", options=atributos)

if st.sidebar.button("Treinar Modelo"): 
    if atributos_selecionados:
        st.sidebar.write("Treinando o modelo com os atributos:", atributos_selecionados)

        x_train_custom = x[atributos_selecionados]
        
        x_train_custom, x_test_custom, y_train_custom, y_test_custom = train_test_split(x_train_custom, y, test_size=0.2, random_state=42)

        model_custom = DecisionTreeClassifier(random_state=42, max_depth=3)
        model_custom.fit(x_train_custom, y_train_custom)

        y_pred_custom = model_custom.predict(x_test_custom)

        st.write("Acurácia (Atributos Selecionados):", accuracy_score(y_test_custom, y_pred_custom))

        plt.figure(figsize=(12, 8), dpi=1600)
        plot_tree(model_custom, filled=True, feature_names=atributos_selecionados, class_names=['Benigno', 'Maligno'], rounded=True, proportion=True)
        plt.title("Árvore de Decisão para Classificação de Câncer de Mama")

        st.pyplot(plt)
        plt.close()
    else:
        st.sidebar.write("Por favor, selecione pelo menos um atributo para treinar o modelo.")
