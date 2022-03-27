import numpy as np
import os
from flask import Flask, request, render_template, make_response
import joblib
import pandas as pd
from PIL import Image

import streamlit as st

model = joblib.load('model/model.pkl')

st.set_page_config(page_title="Random Forest", page_icon="🌱")
st.sidebar.image('static/hmm.png', use_column_width=True)
st.sidebar.title('Menu')
pagSelecionada = st.sidebar.selectbox('Escolha uma seção',['Home','EDA', 'Previsão de vendas', 'Arquitetura do Projeto', 'Desafios', 'Equipe e Agradecimentos'])

if pagSelecionada == 'Home':
    st.title("Bem-vindo!")    
    st.markdown("Este trabalho foi realizado pelo grupo [Random Forest](https://github.com/Chitolina/RandomForestTeam) do curso de treinamento para cientista de dados [Stack Academy](https://www.youtube.com/c/Stack_tecnologias). Os componentes do grupo são: [Alex Graziani](https://github.com/awildt01), [Bruno Freitas](https://github.com/Freitashbruno), [Lucas Chitolina](https://github.com/Chitolina) e [Weber Godoi](https://github.com/webercg).")
    st.markdown("O projeto de estudo escolhido foi o: [Health Insurance Cross Sell Prediction 🏠 🏥](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction).")
    st.markdown("Do que se trata o projeto?")
    st.markdown("Nosso cliente é uma seguradora que forneceu seguro de saúde para seus clientes, agora eles precisam de sua ajuda na construção de um modelo para prever se os segurados (clientes) do ano passado também terão interesse no seguro de veículos fornecido pela empresa.")
    st.markdown("Construir um modelo para prever se um cliente estaria interessado em seguro de veículo é extremamente útil para a empresa, pois ela pode planejar adequadamente sua estratégia de comunicação para alcançar esses clientes e otimizar seu modelo de negócios e receita. Agora, para prever se o cliente estaria interessado em seguro de veículo, você tem informações sobre dados demográficos (gênero, idade, tipo de código de região), veículos (idade do veículo, danos), apólice (premium, canal de fornecimento) etc.")
    st.markdown("Ferramentas utilizadas:")
    st.markdown("● [Análise Exploratória](https://github.com/Chitolina/RandomForestTeam/tree/main/EDA)      ● [Gerenciamento de Tarefas](https://trello.com/b/Nypkyrp3/randomforest)    ● Cloud (em breve)      ● [Modelo Preditivo](https://github.com/Chitolina/RandomForestTeam/blob/main/EDA/Funcao%20de%20Custo%20e%20Selecao%20de%20Modelo.ipynb)     ● [Dashboard BI](https://github.com/Chitolina/RandomForestTeam/tree/main/BI_RANDOMF)    ● Apresentação (em breve)")
    
elif pagSelecionada == 'EDA':
    st.title("Análise Exploratória dos dados")


elif pagSelecionada == 'Previsão de vendas':
    st.title("Cross-Selling: Previsão de vendas")
    st.write("Previsão de vendas de seguro veicular a clientes de seguro de saúde")

    ListaRegiao = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
    ListaCanais = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 163]
    ListaIdadeVeiculo = ["Menos de 1 ano", "Entre 1 e 2 anos", "Mais de 2 anos"]

    sexo = st.radio(
     "Gênero",
     ('Masculino', 'Feminino'))

    if sexo == 'Masculino':
        sexo = 1
    else:
        sexo = 0

    idade = st.number_input(label = 'Idade', min_value=1, max_value=120)
    ##st.slider(label = 'Idade', min_value=1, max_value=180)

    habilitacao = st.radio(
     "Possui Habilitação?",
     ('Sim', 'Não'))

    if habilitacao == 'Sim':
        habilitacao = 1
    else:
        habilitacao = 0

    regiao = st.selectbox('Região',ListaRegiao)

    assegurado_antes = st.radio(
     "Já é assegurado?",
     ('Sim', 'Não'))

    if assegurado_antes == 'Sim':
        assegurado_antes = 1
    else:
        assegurado_antes = 0

    idade_veiculo = st.selectbox('Tempo de uso do veículo',ListaIdadeVeiculo)

    if idade_veiculo == 'Menos de 1 ano':
        idade_veiculo = 0
    elif idade_veiculo == 'Entre 1 e 2 anos':
        idade_veiculo = 1
    elif idade_veiculo == 'Mais de 2 anos':
        idade_veiculo = 2


    veiculo_danificado = st.radio(
     "Veiculo com Sinistro?",
     ('Sim', 'Não'))   

    if veiculo_danificado == 'Sim':
        veiculo_danificado = 1
    else:
        veiculo_danificado = 0

    valorsegurosaude = st.number_input(label = 'Valor anual do seguro de saúde', min_value=1, max_value=800000)

    canalvenda = st.selectbox('Canal de venda',ListaCanais)

    tempofidelidade = st.number_input(label = 'Tempo de fidelidade (Dias)', min_value=1, max_value=800000)


    verificar = st.button("Verificar")

    if verificar:
        Resumo = []
        teste = [sexo,idade,habilitacao,regiao,assegurado_antes,idade_veiculo,veiculo_danificado,valorsegurosaude,canalvenda,tempofidelidade]
        df_api = pd.DataFrame(np.array([teste]), columns=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age','Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']) 
        X = df_api
        classe = int(model.predict(X)[0])

        max_proba = 0
        max_canal = 0


        if classe == 1:
            image = Image.open('static/sim.jpg')
            st.image(image)
        else:
            image = Image.open('static/nao.jpg')
            st.image(image)

        st.write("Avaliando outros canais de venda:")
        for i in ListaCanais:

            df_api['Policy_Sales_Channel'] = i
            X = df_api
            prob1 = model.predict_proba(X)[0][1]

            canal_formatado = '{0:.3g}'.format(i)
            prob1_formatado = prob1*100
            prob1_formatado = '{0:.3g}'.format(prob1_formatado)+"%"

            Resumo.append(np.array([canal_formatado,prob1_formatado]))

            if prob1 > max_proba:
                max_proba = prob1
                max_canal = i

        max_proba = max_proba*100
        max_proba = '{0:.3g}'.format(max_proba)+"%"

        df_resumo = pd.DataFrame(np.array(Resumo), columns=['Canal','Probabilidade']) 
        df_resumo.sort_values(by='Probabilidade',ascending=False, inplace = True)
        st.dataframe(df_resumo.style.highlight_max(axis=0))

        strretorno =  "Melhor canal de venda: "+str(max_canal)+"; Probabilidade: "+str(max_proba)
        stringretorno = str(strretorno)
        st.write(strretorno)

elif pagSelecionada = 'Arquitetura do Projeto'
    st.title("Arquitetura do Projeto")
    
    st.markdown('Para a nosssa engenharia de dados obtamos uma solução em cloud com o Databricks que nos deu a oportunidade de trabalhar com uma technologia basseada em Apache Spark.')
    st.markdown('A organização e o processo das datas (Data Lake) foi feita no Databricks.')
    st.markdown('Criamos uma conta no Github para a disponibilização do Dataset, facilitando o trabalho em grupo e a documentação do projeto')
    img = Image.open("static/diagramm.jpg" ) 
    st.image(img)



elif pagSelecionada == 'Desafios':
    st.title("Principais desafios do projeto")
    
    st.write('O nosso desavio consiste em construir um modelo de previsão dos segurados que teriam interrese em Seguro de Veículos.')
    st.write('Nosso cliente é uma Seguradora que fornece seguro de saúde para seus clientes e que pretende no futuro fornecer também seguro de veículos.')
    st.write('Assim como o seguro médico, existe o seguro de veículo onde todos os anos o cliente precisa pagar um prêmio de certo valor à seguradora para que,em caso de acidente com o veículo, a seguradora forneça uma indenização para o consumidor.')
    st.write('Construir um modelo para prever se um cliente estaria interessado em seguro de veículo é extremamente útil para a empresa, pois ela pode planejar adequadamente sua estratégia de comunicação para alcançar esses clientes e otimizar seu modelo de negócios e receita..')
    st.write('Para prever se o segurado estaria interessado em um  seguro de veículo, temos informações sobre dados demográficos (gênero, idade, tipo de código de região), veículos (idade do veículo, danos), premium e canal de fornecimento.')
    img = Image.open("static/kisspng.jpg" ) 
    st.image(img)

elif pagSelecionada == 'Equipe e Agradecimentos':
    st.title("Equipe e Agradecimentos")
