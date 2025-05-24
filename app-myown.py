import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

houses = pd.read_csv("recife.csv", encoding = "ISO-8859-1")

# Adicionando coluna para aluguel e venda
houses['operation'] = 'sell'
houses.loc[(houses['price'] > 100) & (houses['price'] < 30000),'operation'] = 'rent'

house_sells = houses.loc[houses.operation=='sell']
house_rentals = houses.loc[houses.operation=='rent']

option = st.selectbox(
    'Operation',
     houses['operation'])

'You selected: ', option

st.dataframe(houses)



houses['type'].value_counts().plot(kind='bar', figsize=(8,4))
plt.title('Quantidade de Imóveis por Tipo')
plt.xlabel('Tipo')
plt.ylabel('Contagem')

fig, ax = plt.subplots()
st.pyplot(fig)

houses.plot.scatter(x='area', y='price', alpha=0.5)
plt.title('Área vs Preço')
plt.xlabel('Área (m²)')
plt.ylabel('Preço')

fig, ax = plt.subplots()

st.pyplot(fig)


