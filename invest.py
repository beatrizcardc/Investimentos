import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pygwalker as pyg  # Certifique-se de que está instalado corretamente

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

# Carregar os dados do CSV atualizado
csv_url = 'https://raw.githubusercontent.com/beatrizcardc/TechChallenge2_Otimizacao/main/Pool_Investimentos.csv'
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Erro ao carregar o CSV: {e}")
    st.stop()

# Extrair retornos do CSV para os 34 ativos
retornos_12m = df['Rentabilidade 12 meses'].values
retornos_24m = df['Rentabilidade 24 meses'].values
retornos_36m = df['Rentabilidade 36 meses'].values

# Lista de tickers das 15 ações, criptomoedas e dólar
tickers_acoes_cripto_dolar = ['VALE3.SA', 'PETR4.SA', 'JBSS3.SA', 'MGLU3.SA', 'RENT3.SA',
                              'B3SA3.SA', 'WEGE3.SA', 'EMBR3.SA', 'GOLL4.SA', 'ITUB4.SA',
                              'BTC-USD', 'ADA-USD', 'ETH-USD', 'LTC-USD', 'BRL=X']

# Baixar dados históricos de preços para as 15 ações e criptos
dados_historicos_completos = yf.download(tickers_acoes_cripto_dolar, start='2021-01-01', end='2024-01-01')['Adj Close']
dados_historicos_completos.fillna(dados_historicos_completos.mean(), inplace=True)

# Preencher valores NaN nos dados históricos com a média da coluna correspondente
dados_historicos_completos.fillna(dados_historicos_completos.mean(), inplace=True)

# Calcular os retornos diários e o desvio padrão (volatilidade) anualizado para as 15 ações, criptos e dólar
retornos_diarios_completos = dados_historicos_completos.pct_change().dropna()
riscos_acoes_cripto_dolar = retornos_diarios_completos.std() * np.sqrt(252)  # Riscos anualizados (15 ativos)

# Ajustar riscos para criptomoedas e ativos mais arriscados
risco_cripto = riscos_acoes_cripto_dolar[10:14] * 1.5  # Ponderar mais para os criptoativos
riscos_acoes_cripto_dolar[10:14] = risco_cripto

# Definir riscos assumidos para os ativos de renda fixa e tesouro (totalizando 19 ativos)
riscos_fixa_tesouro = np.array([0.05, 0.06, 0.04, 0.03, 0.04, 0.05, 0.05, 0.05, 0.06, 0.04, 0.05, 0.03, 0.04, 0.06, 0.04, 0.05, 0.03, 0.04, 0.03])

# Combinar os riscos de ações, criptomoedas e renda fixa/tesouro para totalizar 34 ativos
riscos_completos_final = np.concatenate((riscos_acoes_cripto_dolar.values, riscos_fixa_tesouro))

# Exemplo de dados reais para retornos e riscos 
retornos_reais = np.random.rand(34) * 0.4  # Retornos simulados entre 0% e 40%
retornos_ajustados = retornos_reais.copy()
retornos_ajustados[10:14] *= 1.2  # Aumentar em 20% os retornos das criptos
retornos_ajustados[:10] *= 1.15   # Aumentar em 15% os retornos das ações

# Definir qual conjunto de retornos será utilizado com base na escolha do usuário
retornos_usados = retornos_ajustados if tipo_retorno == "Ajustados" else retornos_reais

# Função para calcular o Sharpe Ratio
def calcular_sharpe(portfolio, retornos, riscos, taxa_livre_risco):
    retorno_portfolio = np.dot(portfolio, retornos)
    risco_portfolio = np.sqrt(np.dot(portfolio, riscos ** 2))

    if risco_portfolio < 0.01:
        risco_portfolio = 0.01

    sharpe_ratio = (retorno_portfolio - taxa_livre_risco) / risco_portfolio
    if sharpe_ratio < 1.0:
        sharpe_ratio *= 0.8
    elif sharpe_ratio > 3:
        sharpe_ratio *= 0.2
    return sharpe_ratio

# Função para ajustar as alocações
def ajustar_alocacao(portfolio, limite_max=0.25):
    portfolio = np.clip(portfolio, 0, limite_max)
    portfolio /= portfolio.sum()
    return portfolio

# Função de cruzamento ajustada
def cruzamento(pai1, pai2):
    num_pontos_corte = np.random.randint(1, 4)
    pontos_corte = sorted(np.random.choice(range(1, len(pai1)), num_pontos_corte, replace=False))
    filho1, filho2 = pai1.copy(), pai2.copy()

    if len(pontos_corte) % 2 != 0:
        pontos_corte.append(len(pai1))

    for i in range(0, len(pontos_corte) - 1, 2):
        filho1[pontos_corte[i]:pontos_corte[i+1]] = pai2[pontos_corte[i]:pontos_corte[i+1]]
        filho2[pontos_corte[i]:pontos_corte[i+1]] = pai1[pontos_corte[i]:pontos_corte[i+1]]

    filho1 = ajustar_alocacao(filho1)
    filho2 = ajustar_alocacao(filho2)

    return filho1, filho2

# Função de mutação
def mutacao(portfolio, taxa_mutacao, limite_max=0.25):
    if np.random.random() < taxa_mutacao:
        i = np.random.randint(0, len(portfolio))
        portfolio[i] += np.random.uniform(-0.1, 0.1)
        portfolio = ajustar_alocacao(portfolio, limite_max)
    return portfolio

# Função para gerar a população inicial
def gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, num_ativos):
    populacao = [genoma_inicial]
    for _ in range(num_portfolios - 1):
        populacao.append(np.random.dirichlet(np.ones(num_ativos)))
    return populacao

# Rodar o algoritmo genético
def algoritmo_genetico_com_genoma_inicial(retornos, riscos, genoma_inicial, taxa_livre_risco=0.1075, num_portfolios=100, geracoes=100, usar_elitismo=True, taxa_mutacao=0.05):
    populacao = gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, len(retornos))
    melhor_portfolio = genoma_inicial
    melhor_sharpe = calcular_sharpe(genoma_inicial, retornos, riscos, taxa_livre_risco)
    
    evolucao_sharpe = []  # Armazenar a evolução do Sharpe Ratio em cada geração

    for geracao in range(geracoes):
        fitness_scores = np.array([calcular_sharpe(port, retornos, riscos, taxa_livre_risco) for port in populacao])
        indice_melhor_portfolio = np.argmax(fitness_scores)
        
        # Armazenar o melhor Sharpe Ratio de cada geração
        melhor_sharpe = fitness_scores[indice_melhor_portfolio]
        melhor_portfolio = populacao[indice_melhor_portfolio]
        evolucao_sharpe.append(melhor_sharpe)

        populacao = selecao_torneio(populacao, fitness_scores)
        nova_populacao = []
        for i in range(0, len(populacao), 2):
            pai1, pai2 = populacao[i], populacao[i+1]
            filho1, filho2 = cruzamento(pai1, pai2)
            filho1 = ajustar_alocacao(filho1)
            filho2 = ajustar_alocacao(filho2)
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            nova_populacao.append(mutacao(filho2, taxa_mutacao))

        # Inserir o elitismo
        if usar_elitismo:
            nova_populacao[0] = melhor_portfolio

        populacao = nova_populacao

    return melhor_portfolio, evolucao_sharpe

# Rodar o algoritmo genético com o genoma inicial fixo
melhor_portfolio, evolucao_sharpe = algoritmo_genetico_com_genoma_inicial(
    retornos_usados,  # Usar o conjunto de retornos selecionado pelo usuário
    riscos_completos_final,  # Usar a variável de riscos correta
    genoma_inicial,  # Genoma inicial
    taxa_livre_risco,  # Taxa livre de risco
    num_portfolios=100,  # Número de portfólios
    geracoes=100,  # Número de gerações
    usar_elitismo=usar_elitismo,  # Definido pelo usuário
    taxa_mutacao=taxa_mutacao  # Definido pelo usuário
)

# Mostrar a evolução do Sharpe Ratio em um gráfico após a tabela de portfólio
st.write("Evolução do Sharpe Ratio ao longo das gerações:")

# Mostrar a evolução do Sharpe Ratio em um gráfico
fig, ax = plt.subplots()
ax.plot(range(1, len(evolucao_sharpe) + 1), evolucao_sharpe, label='Sharpe Ratio', marker='o')
ax.set_xlabel('Gerações')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Evolução do Sharpe Ratio ao longo das gerações')
ax.legend()

# Exibir o gráfico no Streamlit
st.pyplot(fig)

# Distribuir o valor total de investimento entre os ativos com base na melhor alocação
distribuicao_investimento = melhor_portfolio * valor_total
distribuicao_df = pd.DataFrame({
    'Ativo': df['Ativo'].values,
    'Alocacao (%)': melhor_portfolio * 100,
    'Valor Investido (R$)': distribuicao_investimento
})

# Ordenar o DataFrame pela coluna 'Alocacao (%)' em ordem decrescente
distribuicao_df = distribuicao_df.sort_values(by='Alocacao (%)', ascending=False)

# Exibir a distribuição ideal do investimento no Streamlit
st.write("Distribuição ideal de investimento (ordenada por alocação):")
st.dataframe(distribuicao_df.style.format({'Alocacao (%)': '{:.2f}', 'Valor Investido (R$)': '{:.2f}'}))

# Função para salvar o DataFrame em um novo CSV para download
csv = distribuicao_df.to_csv(index=False)

# Botão para download do CSV atualizado
st.download_button(label="Baixar CSV Atualizado", data=csv, file_name='Pool_Investimentos_Atualizacao2.csv', mime='text/csv')

# Calcular os retornos esperados com base nas alocações
retorno_12m = np.dot(melhor_portfolio, retornos_12m)
retorno_24m = np.dot(melhor_portfolio, retornos_24m)
retorno_36m = np.dot(melhor_portfolio, retornos_36m)

# Exibir os retornos esperados no Streamlit
st.write(f"Retorno esperado em 12 meses: {retorno_12m:.2f}%")
st.write(f"Retorno esperado em 24 meses: {retorno_24m:.2f}%")
st.write(f"Retorno esperado em 36 meses: {retorno_36m:.2f}%")





