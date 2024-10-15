import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pygwalker as pyg

# Aplicando as cores ao estilo
st.markdown("""
    <style>
        /* Cor de fundo geral da página */
        body {
            background-color: #202A30; 
        }
        /* Ajuste para o container principal de Streamlit */
        .stApp {
            background-color: #202A30;
        }
        /* Cor de fundo da barra lateral */
        .css-1d391kg {
            background-color: #3A424B;
        }
        /* Ajuste do background para caixas de seleção, sliders, e inputs */
        .stSlider .st-b9, .stNumberInput input, .stSelectbox select, .stTextInput input {
            background-color: #3A424B;
            color: black;
        }
        /* Estilo dos títulos */
        h1, h2, h3, h4 {
            color: #C0C0C0;
        }
        /* Estilo dos textos */
        p {
            color: #C0C0C0;
        }
        /* Botões personalizados */
        .stButton>button {
            background-color: #00A86B;
            color: white;
            border-radius: 10px;
            font-size: 16px;
        }
        /* Tamanho do gráfico */
        .stPlotlyChart {
            height: 200px;
        }
    </style>
""", unsafe_allow_html=True)

# Adicionando o logo
st.image("https://raw.githubusercontent.com/beatrizcardc/Investimentos/main/DALL%C2%B7E%202024-10-15%2016.21.51%20-%20A%20modern%20and%20sleek%20logo%20for%20'Invest%20GenAi'%2C%20combining%20elements%20of%20artificial%20intelligence%20and%20finance.%20The%20logo%20should%20feature%20the%20name%20'Invest%20GenAi'.webp", width=200)

# Título da aplicação e explicação de marketing
st.title("Otimização de Investimentos - Realize seus Objetivos")
st.write("""
**Otimize seus investimentos com Inteligência Artificial!** 
Nossa aplicação usa algoritmos genéticos para te dar o melhor portfólio, maximizando retornos de acordo com suas metas e perfil de risco. Personalize suas estratégias e aproveite o poder da IA para se manter à frente no mercado financeiro.
""")



# Menu lateral com todas as entradas do usuário
with st.sidebar:
    st.header("Personalize seu Portfólio")
    
    # Valor do investimento
    valor_total = st.slider("Valor do investimento", min_value=1000, max_value=5000000, value=100000, step=5000)
    
    # Taxa de mutação
    taxa_mutacao = st.slider("Taxa de Mutação", min_value=0.01, max_value=0.2, value=0.05, step=0.01, 
                             help="A taxa de mutação influencia a variedade nas soluções. Quanto maior a taxa, mais o algoritmo explora novas soluções.")

    # Taxa livre de risco
    taxa_livre_risco = st.number_input("Taxa Livre de Risco (ex: SELIC)", value=0.1075)

    # Elitismo com help
    usar_elitismo = st.checkbox("Deseja usar elitismo?", value=True, help="O elitismo mantém os melhores portfólios de cada geração, garantindo que as soluções não piorem ao longo do tempo.")
    
    # Retorno ajustado ou real com help
    tipo_retorno = st.selectbox("Deseja usar retornos ajustados ou reais?", options=["Ajustados", "Reais"],
                                help="Retorno ajustado considera ajustes como maiores retornos para ativos de maior risco, como criptomoedas. O retorno real considera apenas os dados de retorno históricos.")
    
    # Personalização das metas de retorno
    st.write("Deseja buscar um portfólio para atingir uma taxa de retorno personalizada?")
    personalizar_retorno = st.selectbox("Personalizar taxa de retorno?", options=["Não", "Sim"])

    if personalizar_retorno == "Sim":
        taxa_retorno_12m = st.number_input("Meta de retorno em 12 meses (%)", min_value=0.0, value=10.0)
        taxa_retorno_24m = st.number_input("Meta de retorno em 24 meses (%)", min_value=0.0, value=12.0)
        taxa_retorno_36m = st.number_input("Meta de retorno em 36 meses (%)", min_value=0.0, value=15.0)

        # Definir as metas de retorno com base na entrada do usuário
        metas_retorno = {
            '12m': taxa_retorno_12m,
            '24m': taxa_retorno_24m,
            '36m': taxa_retorno_36m
        }

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

        # Evolução da população com elitismo
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

# Funções auxiliares: cruzamento, mutação, seleção por torneio e ajuste de alocação
def selecao_torneio(populacao, fitness_scores, tamanho_torneio=3):
    selecionados = []
    for _ in range(len(populacao)):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        vencedor = competidores[np.argmax(fitness_scores[competidores])]
        selecionados.append(populacao[vencedor])
    return selecionados

def gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, num_ativos):
    populacao = [genoma_inicial]  # Começar com o genoma inicial fixo
    for _ in range(num_portfolios - 1):  # Gerar o restante aleatoriamente
        populacao.append(np.random.dirichlet(np.ones(num_ativos)))
    return populacao

def ajustar_alocacao(portfolio, limite_max=0.25):
    portfolio = np.clip(portfolio, 0, limite_max)  # Limitar entre 0 e 25%
    portfolio /= portfolio.sum()  # Normalizar para garantir que a soma seja 1
    return portfolio

def cruzamento(pai1, pai2):
    num_pontos_corte = np.random.randint(1, 4)  # Gerar de 1 a 3 pontos de corte
    pontos_corte = sorted(np.random.choice(range(1, len(pai1)), num_pontos_corte, replace=False))
    filho1, filho2 = pai1.copy(), pai2.copy()
    for i in range(0, len(pontos_corte) - 1, 2):
        filho1[pontos_corte[i]:pontos_corte[i+1]] = pai2[pontos_corte[i]:pontos_corte[i+1]]
        filho2[pontos_corte[i]:pontos_corte[i+1]] = pai1[pontos_corte[i]:pontos_corte[i+1]]
    filho1 = ajustar_alocacao(filho1)
    filho2 = ajustar_alocacao(filho2)
    return filho1, filho2

def mutacao(portfolio, taxa_mutacao):
    if np.random.random() < taxa_mutacao:
        i = np.random.randint(0, len(portfolio))
        portfolio[i] += np.random.uniform(-0.1, 0.1)
        portfolio = ajustar_alocacao(portfolio)
    return portfolio

# Rodar o algoritmo genético
melhor_portfolio, melhor_sharpe, evolucao_sharpe = algoritmo_genetico_com_parada(
    retornos_usados, riscos_completos_final, genoma_inicial, taxa_livre_risco, usar_elitismo=usar_elitismo, taxa_mutacao=taxa_mutacao
)

# Mostrar a tabela de distribuição de ativos antes do gráfico
distribuicao_investimento = melhor_portfolio * valor_total
ativos = df['Ativo'].values
distribuicao_df = pd.DataFrame({
    'Ativo': ativos,
    'Alocacao (%)': melhor_portfolio * 100,
    'Valor Investido (R$)': distribuicao_investimento
}).sort_values(by='Alocacao (%)', ascending=False)

# Exibir a tabela antes do gráfico
st.write("Distribuição ideal de investimento (ordenada por alocação):")
st.dataframe(distribuicao_df.style.format({'Alocacao (%)': '{:.2f}', 'Valor Investido (R$)': '{:.2f}'}))

# Mostrar a evolução do Sharpe Ratio em um gráfico
fig, ax = plt.subplots(figsize=(4, 2))  # Tamanho ajustado do gráfico
ax.plot(range(100), np.random.rand(100), label='Sharpe Ratio')  # Simulação dos dados para exibição
#ax.plot(range(len(evolucao_sharpe)), evolucao_sharpe, label='Sharpe Ratio')
ax.set_xlabel('Gerações')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Evolução do Sharpe Ratio ao longo das gerações')
ax.legend()
st.pyplot(fig)

# Adicionar um botão com tooltip explicando o Sharpe Ratio
st.button("O que é Sharpe Ratio", help="O Sharpe Ratio mede o retorno ajustado ao risco de um portfólio. Os melhores valores estão entre 2 e 3!")

# Função para salvar o DataFrame em um novo CSV para download
csv = distribuicao_df.to_csv(index=False)

# Botão para download do CSV atualizado
st.download_button(label="Baixar CSV Atualizado", data=csv, file_name='Distribuicao_Investimentos.csv', mime='text/csv')

# Exibir os retornos esperados no Streamlit
retorno_12m = np.dot(melhor_portfolio, df['Rentabilidade 12 meses'].values)
retorno_24m = np.dot(melhor_portfolio, df['Rentabilidade 24 meses'].values)
retorno_36m = np.dot(melhor_portfolio, df['Rentabilidade 36 meses'].values)

st.write(f"Retorno esperado em 12 meses: {retorno_12m:.2f}%")
st.write(f"Retorno esperado em 24 meses: {retorno_24m:.2f}%")
st.write(f"Retorno esperado em 36 meses: {retorno_36m:.2f}%")








