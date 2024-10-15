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









