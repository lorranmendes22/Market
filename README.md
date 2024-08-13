

# Classificação de Marketing em Investimentos

## Visão Geral

Este projeto tem como objetivo prever se os clientes de um banco aplicarão seu dinheiro em investimentos com base em uma campanha de marketing. Utilizamos técnicas de machine learning para construir e avaliar modelos de classificação. O projeto abrange desde a leitura e análise dos dados até o ajuste, avaliação e comparação dos modelos.

## Estrutura do Projeto

1. **Leitura dos Dados**
   - Os dados são carregados a partir de um arquivo CSV utilizando a biblioteca `pandas`.
   - O dataset é composto por 1268 registros e 9 colunas.

2. **Análise Exploratória**
   - Exploramos as variáveis categóricas e numéricas usando a biblioteca `plotly` para identificar padrões e inconsistências nos dados.

3. **Transformação dos Dados**
   - **Variáveis Explicativas**: Transformamos variáveis categóricas em um formato numérico usando `OneHotEncoder` e mantemos as variáveis numéricas como estão.
   - **Variável Alvo**: Transformamos a variável alvo em valores numéricos binários com `LabelEncoder`.

4. **Divisão dos Dados**
   - Os dados são divididos em conjuntos de treinamento e teste com `train_test_split`.

5. **Ajuste e Avaliação dos Modelos**
   - **Modelo Base**: Utilizamos `DummyClassifier` para estabelecer uma linha de base.
   - **Árvore de Decisão**: Implementamos um modelo de árvore de decisão (`DecisionTreeClassifier`) e avaliamos seu desempenho.
   - **KNN**: Ajustamos um modelo de K-Nearest Neighbors (`KNeighborsClassifier`) e normalizamos os dados com `MinMaxScaler`.

6. **Escolha e Salvamento do Melhor Modelo**
   - Comparamos o desempenho dos modelos e escolhemos o melhor.
   - O modelo com melhor desempenho é salvo em um arquivo usando `pickle` para uso futuro.

## Passos para Executar o Projeto

1. **Instalação de Dependências**
   Instale as bibliotecas necessárias usando o comando:
   ```bash
   pip install pandas plotly scikit-learn matplotlib
   ```

2. **Carregar e Explorar os Dados**
   - O arquivo CSV é carregado e analisado para verificar a integridade dos dados e explorar variáveis categóricas e numéricas.

3. **Transformar os Dados**
   - Realize a transformação das variáveis explicativas e da variável alvo para o formato numérico adequado.

4. **Dividir os Dados e Ajustar Modelos**
   - Divida os dados em conjuntos de treinamento e teste.
   - Ajuste e avalie os modelos de classificação.

5. **Comparar Modelos e Salvar o Melhor**
   - Compare o desempenho dos modelos e salve o modelo com melhor desempenho.

## Código

Aqui está o código utilizado no projeto:

```python
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle

# Carregar os dados
url = "https://raw.githubusercontent.com/lorranmendes22/Market/main/marketing.csv"
dados = pd.read_csv(url)

# Verificar dados nulos e tipos de dados
dados.info()

# Explorando os dados
px.histogram(dados, x='aderencia_investimento', text_auto=True)
px.histogram(dados, x='estado_civil', text_auto=True, color='aderencia_investimento', barmode='group')
px.box(dados, x='idade', color='aderencia_investimento')

# Transformar variáveis explicativas e variável alvo
y = dados['aderencia_investimento']
x = dados.drop('aderencia_investimento', axis=1)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['estado_civil', 'escolaridade', 'inadimplencia', 'fez_emprestimo']),
        ('num', 'passthrough', ['idade', 'saldo', 'tempo_ult_contato', 'numero_contatos'])
    ]
)
x_transformed = preprocessor.fit_transform(x)
label_encoder = LabelEncoder()
y_transformed = label_encoder.fit_transform(y)

# Dividir os dados entre treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y_transformed, test_size=0.2, random_state=42)

# Ajustar e avaliar modelos
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(x_train, y_train)
print(f"Desempenho do modelo base: {dummy.score(x_test, y_test)}")

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
print(f"Desempenho da Árvore de Decisão: {tree.score(x_test, y_test)}")
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=preprocessor.get_feature_names_out())
plt.show()

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)
print(f"Desempenho do KNN: {knn.score(x_test_scaled, y_test)}")

# Escolher e salvar o melhor modelo
model_scores = {
    'Modelo Base': dummy.score(x_test, y_test),
    'Árvore de Decisão': tree.score(x_test, y_test),
    'KNN': knn.score(x_test_scaled, y_test)
}
melhor_modelo_nome = max(model_scores, key=model_scores.get)
melhor_modelo = {'Modelo Base': dummy, 'Árvore de Decisão': tree, 'KNN': knn}[melhor_modelo_nome]
print(f"Melhor modelo: {melhor_modelo_nome} com precisão de {model_scores[melhor_modelo_nome]}")
with open('melhor_modelo.pkl', 'wb') as f:
    pickle.dump(melhor_modelo, f)

# Carregar o modelo salvo
with open('melhor_modelo.pkl', 'rb') as f:
    modelo_carregado = pickle.load(f)
print("Modelo carregado com sucesso!")
```

## Conclusão

Este projeto forneceu uma abordagem prática para análise de dados e construção de modelos de machine learning para prever a aderência a investimentos em uma campanha de marketing. O modelo KNN apresentou o melhor desempenho e foi salvo para uso futuro.

