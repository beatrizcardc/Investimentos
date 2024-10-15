import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pygwalker as pyg

# Título da aplicação e explicação
st.title("Otimização de Investimentos - Realize seus Objetivos")
st.write("Esta aplicação utiliza algoritmos genéticos para otimizar portfólios de investimentos, buscando maximizar o Sharpe Ratio com base em uma série de parâmetros que você pode personalizar.")

# Menu lateral com todas as entradas do usuário
with st.sidebar:
    st.header("Personalize seu Portfólio")
    
    # Valor do investimento
    valor_total = st.slider("Valor do investimento", min_value=1000, max_value=5000000, value=100000, step=5000)
    
    # Taxa de mutação
    taxa_mutacao = st.slider("Taxa de Mutação", min_value=0.01, max_value=0.2, value=0.05, step=0.01, 
                             help="A taxa de mutação influencia a variedade nas soluções.")
    
    # Taxa livre de risco
    taxa_livre_risco = st.number_input("Taxa Livre de Risco (ex: SELIC)", value=0.1075)
    
    # Elitismo
    usar_elitismo = st.checkbox("Deseja usar elitismo?", value=True)
    
    # Retorno ajustado ou real
    tipo_retorno = st.selectbox("Deseja usar retornos ajustados ou reais?", options=["Ajustados", "Reais"])
    
    # Explicação do Sharpe Ratio
    st.write("O **Sharpe Ratio** mede o retorno ajustado ao risco de um portfólio. Quanto maior, melhor.")

# Carregar dados do CSV
csv_url = 'https://raw.githubusercontent.com/beatrizcardc/TechChallenge2_Otimizacao/main/Pool_Investimentos.csv'
df = pd.read_csv(csv_url)

# Lista de tickers das 15 ações, criptomoedas e dólar
tickers_acoes_cripto_dolar = ['VALE3.SA', 'PETR4.SA', 'JBSS3.SA', 'MGLU3.SA', 'RENT3.SA',
                              'B3SA3.SA', 'WEGE3.SA', 'EMBR3.SA', 'GOLL4.SA', 'ITUB4.SA',
                              'BTC-USD', 'ADA-USD', 'ETH-USD', 'LTC-USD', 'BRL=X']

# Baixar dados históricos de preços para as 15 ações e criptos
dados_historicos_completos = yf.download(tickers_acoes_cripto_dolar, start='2021-01-01', end='2024-01-01')['Adj Close']

# Preencher valores NaN nos dados históricos com a média da coluna correspondente
dados_historicos_completos.fillna(dados_historicos_completos.mean(), inplace=True)

# Calcular os retornos diários e o desvio padrão (volatilidade) anualizado para as 15 ações, criptos e dólar
retornos_diarios_completos = dados_historicos_completos.pct_change().dropna()
riscos_acoes_cripto_dolar = retornos_diarios_completos.std() * np.sqrt(252)  # Riscos anualizados (15 ativos)

# Ajustar riscos para criptomoedas e ativos mais arriscados
risco_cripto = riscos_acoes_cripto_dolar[10:14] * 1.5  # Ponderar mais para os criptoativos (Bitcoin, Cardano, Ethereum, Litecoin)

# Atualizar os riscos das criptomoedas com o novo valor ponderado
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

# Função para gerar o genoma inicial de portfólios com 34 ativos
genoma_inicial = np.array([
    0.00, 0.00, 0.20, 0.00, 0.05, 0.00, 0.03, 0.00, 0.00, 0.03,
    0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.06,
    0.10, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.05, 0.00, 0.05,
    0.05, 0.03, 0.05, 0.00
])

# Função para calcular o Sharpe Ratio com penalização e normalização
def calcular_sharpe(portfolio, retornos, riscos, taxa_livre_risco):
    retorno_portfolio = np.dot(portfolio, retornos)  # Retorno ponderado
    risco_portfolio = np.sqrt(np.dot(portfolio, riscos ** 2))  # Risco ponderado

    # Evitar divisões por zero ou risco muito baixo
    if risco_portfolio < 0.01:
        risco_portfolio = 0.01

    # Calcular o Sharpe Ratio
    sharpe_ratio = (retorno_portfolio - taxa_livre_risco) / risco_portfolio

    # Penalizar Sharpe Ratios muito baixos ou incentivar os maiores
    if sharpe_ratio < 1.0:
        sharpe_ratio = sharpe_ratio * 0.8
    elif sharpe_ratio > 3:
        sharpe_ratio = sharpe_ratio * 0.2

    return sharpe_ratio

# Função para rodar o algoritmo genético com ajustes de penalidade e variabilidade
def algoritmo_genetico_com_parada(retornos, riscos, genoma_inicial, taxa_livre_risco=0.1075, num_portfolios=100, geracoes=100, usar_elitismo=True, taxa_mutacao=0.05):
    populacao = gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, len(retornos))
    melhor_portfolio = genoma_inicial
    melhor_sharpe = calcular_sharpe(genoma_inicial, retornos, riscos, taxa_livre_risco)
    geracoes_sem_melhoria = 0
    evolucao_sharpe = []

    for geracao in range(geracoes):
        fitness_scores = np.array([calcular_sharpe(port, retornos, riscos, taxa_livre_risco) for port in populacao])
        indice_melhor_portfolio = np.argmax(fitness_scores)
        if fitness_scores[indice_melhor_portfolio] > melhor_sharpe:
            melhor_sharpe = fitness_scores[indice_melhor_portfolio]
            melhor_portfolio = populacao[indice_melhor_portfolio]
            geracoes_sem_melhoria = 0
        else:
            geracoes_sem_melhoria += 1
        
        if melhor_sharpe >= 3 or geracoes_sem_melhoria >= 5:
            break

        populacao = selecao_torneio(populacao, fitness_scores)
        nova_populacao = []
        for i in range(0, len(populacao), 2):
            pai1, pai2 = populacao[i], populacao[i+1]
            filho1, filho2 = cruzamento(pai1, pai2)
            filho1 = ajustar_alocacao(filho1)
            filho2 = ajustar_alocacao(filho2)
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            nova_populacao.append(mutacao(filho2, taxa_mutacao))

        if usar_elitismo:
            nova_populacao[0] = melhor_portfolio

        populacao = nova_populacao
        evolucao_sharpe.append(melhor_sharpe)

    return melhor_portfolio, melhor_sharpe, evolucao_sharpe

# Função para verificar se o portfólio atende às metas de retorno
def verificar_retorno(portfolio, retornos_12m, retornos_24m, retornos_36m, metas_retorno):
    retorno_portfolio_12m = np.dot(portfolio, retornos_12m)
    retorno_portfolio_24m = np.dot(portfolio, retornos_24m)
    retorno_portfolio_36m = np.dot(portfolio, retornos_36m)
    
    if (retorno_portfolio_12m >= metas_retorno['12m'] and
        retorno_portfolio_24m >= metas_retorno['24m'] and
        retorno_portfolio_36m >= metas_retorno['36m']):
        return True
    return False

# Oferecer a opção para o usuário definir metas de retorno personalizadas
st.write("Deseja buscar um portfólio para atingir uma taxa de retorno personalizada?")
personalizar_retorno = st.selectbox("Personalizar taxa de retorno?", options=["Não", "Sim"])

# Se o usuário escolher 'Sim', permitir a entrada de metas de retorno
if personalizar_retorno == "Sim":
    taxa_retorno_12m = st.number_input("Meta de retorno em 12 meses (%)", min_value=0.0, value=10.0)
    taxa_retorno_24m = st.number_input("Meta de retorno em 24 meses (%)", min_value=0.0, value=12.0)
    taxa_retorno_36m = st.number_input("Meta de retorno em 36 meses (%)", min_value=0.0, value=15.0)

    metas_retorno = {
        '12m': taxa_retorno_12m,
        '24m': taxa_retorno_24m,
        '36m': taxa_retorno_36m
    }

    melhor_portfolio = None
    for geracao in range(100):
        populacao = gerar_portfolios_com_genoma_inicial(genoma_inicial, 100, len(retornos_usados))
        for portfolio in populacao:
            if verificar_retorno(portfolio, retornos_12m, retornos_24m, retornos_36m, metas_retorno):
                melhor_portfolio = portfolio
                break
        if melhor_portfolio is not None:
            break

    if melhor_portfolio is not None:
        distribuicao_investimento = melhor_portfolio * valor_total
        distribuicao_df = pd.DataFrame({
            'Ativo': df['Ativo'].values,
            'Alocacao (%)': melhor_portfolio * 100,
            'Valor Investido (R$)': distribuicao_investimento
        }).sort_values(by='Alocacao (%)', ascending=False)
        
        pyg.walk(distribuicao_df)

        retorno_12m = np.dot(melhor_portfolio, retornos_12m)
        retorno_24m = np.dot(melhor_portfolio, retornos_24m)
        retorno_36m = np.dot(melhor_portfolio, retornos_36m)

        st.write(f"Novo retorno esperado em 12 meses: {retorno_12m:.2f}%")
        st.write(f"Novo retorno esperado em 24 meses: {retorno_24m:.2f}%")
        st.write(f"Novo retorno esperado em 36 meses: {retorno_36m:.2f}%")

        csv = distribuicao_df.to_csv(index=False)
        st.download_button(label="Baixar CSV Atualizado", data=csv, file_name='Distribuicao_Investimento.csv', mime='text/csv')
else:
    st.write("Você optou por não personalizar as metas de retorno.")






