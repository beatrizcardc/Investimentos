import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pygwalker as pyg

# Título da aplicação no centro da tela
st.title("Otimização de Investimentos - Realize seus Objetivos")

# Menu lateral para personalização
st.sidebar.header("Personalização de Investimento")

# Usar slider bar para selecionar o valor total do investimento
valor_total = st.sidebar.slider(
    "Valor total do investimento", min_value=1000, max_value=1000000, value=100000, step=1000
)

# Adicionar controle para selecionar a taxa de mutação
taxa_mutacao = st.sidebar.slider(
    "Taxa de Mutação", min_value=0.01, max_value=0.2, value=0.05, step=0.01,
    help="A taxa de mutação garante exploração de novas soluções em algoritmos genéticos."
)

# Adicionar controle para a taxa livre de risco
taxa_livre_risco = st.sidebar.number_input(
    "Taxa Livre de Risco (Ex: SELIC, POUPANÇA)", value=0.1075,
    help="Insira uma taxa livre de risco, como a SELIC."
)

# Perguntar sobre o uso de elitismo
usar_elitismo = st.sidebar.selectbox("Deseja usar elitismo?", options=["Sim", "Não"])
usar_elitismo = True if usar_elitismo == "Sim" else False

# Selecionar qual tipo de retorno usar
tipo_retorno = st.sidebar.selectbox("Deseja usar retornos ajustados ou reais?", options=["Ajustados", "Reais"])

# Exibir o valor total de investimento escolhido na tela principal
st.write(f"Você deseja investir: R$ {valor_total}")

# Aqui, implementaríamos o resto da lógica de cálculo e visualização, usando o pygwalker para o front

# Exemplo de integração do pygwalker:
pyg.walk(distribuicao_df)  # Chamar o dataframe que deseja visualizar de maneira interativa com pygwalker

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pygwalker as pyg

# Título da aplicação no centro da tela
st.title("Otimização de Investimentos - Realize seus Objetivos")

# Menu lateral para personalização
st.sidebar.header("Personalização de Investimento")

# Usar slider bar para selecionar o valor total do investimento
valor_total = st.sidebar.slider(
    "Valor total do investimento", min_value=1000, max_value=1000000, value=100000, step=1000
)

# Adicionar controle para selecionar a taxa de mutação
taxa_mutacao = st.sidebar.slider(
    "Taxa de Mutação", min_value=0.01, max_value=0.2, value=0.05, step=0.01,
    help="A taxa de mutação garante exploração de novas soluções em algoritmos genéticos."
)

# Adicionar controle para a taxa livre de risco
taxa_livre_risco = st.sidebar.number_input(
    "Taxa Livre de Risco (Ex: SELIC, POUPANÇA)", value=0.1075,
    help="Insira uma taxa livre de risco, como a SELIC."
)

# Perguntar sobre o uso de elitismo
usar_elitismo = st.sidebar.selectbox("Deseja usar elitismo?", options=["Sim", "Não"])
usar_elitismo = True if usar_elitismo == "Sim" else False

# Selecionar qual tipo de retorno usar
tipo_retorno = st.sidebar.selectbox("Deseja usar retornos ajustados ou reais?", options=["Ajustados", "Reais"])

# Exibir o valor total de investimento escolhido na tela principal
st.write(f"Você deseja investir: R$ {valor_total}")

# Aqui, implementaríamos o resto da lógica de cálculo e visualização, usando o pygwalker para o front

# Exemplo de integração do pygwalker:
# pyg.walk(distribuicao_df)  # Chamar o dataframe que deseja visualizar de maneira interativa com pygwalker


