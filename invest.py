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
            background-color: #202A30;
        }
        /* Ajuste do background para caixas de seleção, sliders, e inputs */
        .stSlider .st-b9, .stNumberInput input, .stSelectbox select, .stTextInput input {
            background-color: ##202A30;
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
st.image("https://raw.githubusercontent.com/beatrizcardc/Investimentos/main/DALL%C2%B7E%202024-10-15%2016.21.51%20-%20A%20modern%20and%20sleek%20logo%20for%20'Invest%20GenAi'%2C%20combining%20elements%20of%20artificial%20intelligence%20and%20finance.%20The%20logo%20should%20feature%20the%20name%20'Invest%20GenAi'.webp", width=150)


# Título da aplicação e explicação de marketing
st.title("Otimização de Investimentos - Realize seus Objetivos")
st.write("""
**Otimize seus investimentos com Inteligência Artificial!** 
Nossa aplicação usa algoritmos genéticos para ajustar automaticamente seu portfólio, maximizando retornos de acordo com suas metas e perfil de risco. Personalize suas estratégias e aproveite o poder da IA para se manter à frente no mercado financeiro.
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

# Função para rodar o algoritmo genético com parada dinâmica
def algoritmo_genetico_com_parada_dinamica(retornos, riscos, genoma_inicial, taxa_livre_risco=0.1075, num_portfolios=100, geracoes=100, usar_elitismo=True, taxa_mutacao=0.05, max_sem_melhoria=20, target_sharpe=3.0):
    populacao = gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, len(retornos))
    melhor_portfolio = genoma_inicial
    melhor_sharpe = calcular_sharpe(genoma_inicial, retornos, riscos, taxa_livre_risco)
    geracoes_sem_melhoria = 0
    evolucao_sharpe = []

    for geracao in range(geracoes):
        # Calcula o Sharpe Ratio para cada portfólio na população
        fitness_scores = np.array([calcular_sharpe(port, retornos, riscos, taxa_livre_risco) for port in populacao])
        
        # Encontra o melhor portfólio da geração atual
        indice_melhor_portfolio = np.argmax(fitness_scores)
        melhor_sharpe_da_geracao = fitness_scores[indice_melhor_portfolio]
        
        # Atualiza o melhor Sharpe Ratio se houver melhoria
        if melhor_sharpe_da_geracao > melhor_sharpe:
            melhor_sharpe = melhor_sharpe_da_geracao
            melhor_portfolio = populacao[indice_melhor_portfolio]
            geracoes_sem_melhoria = 0  # Reinicia o contador de gerações sem melhoria
        else:
            geracoes_sem_melhoria += 1  # Aumenta o contador se não houver melhoria
        
        # Adiciona o Sharpe Ratio da geração atual à lista de evolução
        evolucao_sharpe.append(melhor_sharpe)

        # Critério de parada dinâmica
        if melhor_sharpe >= target_sharpe:
            st.write(f"Parou porque o Sharpe Ratio atingiu {melhor_sharpe:.2f} na geração {geracao+1}.")
            break
        elif geracoes_sem_melhoria >= max_sem_melhoria:
            st.write(f"Parou porque não houve melhoria por {max_sem_melhoria} gerações consecutivas.")
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

# Executar o algoritmo com a nova lógica de parada dinâmica
retornos_usados = np.random.rand(34) * 0.4  # Simulação de dados
riscos_completos_final = np.random.rand(34) * 0.2  # Simulação de riscos
genoma_inicial = np.array([0.05] * 34)  # Genoma inicial com alocações iguais

melhor_portfolio, melhor_sharpe, evolucao_sharpe = algoritmo_genetico_com_parada_dinamica(
    retornos_usados, riscos_completos_final, genoma_inicial, taxa_livre_risco=0.1075,
    num_portfolios=100, geracoes=100, usar_elitismo=True, taxa_mutacao=0.05,
    max_sem_melhoria=20, target_sharpe=3.0
)

# Exibir gráfico da evolução do Sharpe Ratio
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(evolucao_sharpe)), evolucao_sharpe, label='Sharpe Ratio')
ax.set_xlabel('Gerações')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Evolução do Sharpe Ratio ao longo das gerações')
ax.legend()
st.pyplot(fig)









