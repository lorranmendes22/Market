# README: Análise e Previsão de Aderência de Investimentos

## Descrição do Projeto

Este projeto visa analisar dados de uma campanha de marketing para prever se clientes de um banco irão investir seu dinheiro. Usando técnicas de machine learning, o objetivo é construir um modelo de classificação que possa prever a aderência dos clientes aos investimentos com base em suas características.

## Conteúdo

1. **Introdução**
2. **Leitura dos Dados**
3. **Exploração dos Dados**
4. **Transformação dos Dados**
5. **Ajuste de Modelos**
6. **Comparação de Modelos**
7. **Salvamento do Melhor Modelo**
8. **Uso do Modelo**

## 1. Introdução

O projeto utiliza dados de uma campanha de marketing para prever se clientes irão investir ou não. Utilizamos técnicas de machine learning para construir e avaliar modelos de classificação.

## 2. Leitura dos Dados

Os dados são carregados a partir de um arquivo CSV usando a biblioteca pandas. Aqui está como fazemos isso:

```python
import pandas as pd

# Carregar o dataset
url = "https://raw.githubusercontent.com/lorranmendes22/Investimento-Market/main/marketing_investimento.csv"
dados = pd.read_csv(url)
dados.head()
```

## 3. Exploração dos Dados

A análise exploratória ajuda a entender a distribuição dos dados e identificar padrões e inconsistências. Utilizamos a biblioteca plotly para visualização:

```python
import plotly.express as px

# Histogramas das variáveis categóricas
px.histogram(dados, x='aderencia_investimento', text_auto=True)
px.histogram(dados, x='estado_civil', text_auto=True, color='aderencia_investimento', barmode='group')

# Boxplot das variáveis numéricas
px.box(dados, x='idade', color='aderencia_investimento')
```

## 4. Transformação dos Dados

Para usar algoritmos de machine learning, os dados precisam estar no formato numérico. Transformamos as variáveis categóricas e a variável alvo da seguinte maneira:

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Separação das variáveis explicativas e variável alvo
x = dados.drop(['fez_emprestimo', 'inadimplencia', 'aderencia_investimento'], axis=1)
y = dados['aderencia_investimento']

# Transformação das variáveis explicativas
transformer = make_column_transformer(
    (StandardScaler(), x.select_dtypes(include=['number']).columns),
    (OneHotEncoder(), x.select_dtypes(include=['object']).columns)
)
x_transformed = transformer.fit_transform(x)

# Transformação da variável alvo
le = LabelEncoder()
y_transformed = le.fit_transform(y)
```

## 5. Ajuste de Modelos

Os dados são divididos em conjuntos de treino e teste. Modelos de machine learning são ajustados e avaliados para prever a aderência de investimentos:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(x_transformed, y_transformed, test_size=0.2, random_state=42)

# Treinamento do modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo:", accuracy)
```

## 6. Comparação de Modelos

Comparação de diferentes modelos de classificação, incluindo Dummy Classifier, Árvore de Decisão e KNN:

```python
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Dummy Classifier
dummy = DummyClassifier()
dummy.fit(X_train, y_train)
dummy_accuracy = dummy.score(X_test, y_test) * 100

# Árvore de Decisão
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_accuracy = tree.score(X_test, y_test) * 100

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_accuracy = knn.score(X_test, y_test) * 100

print(f"Acurácia do Dummy Classifier: {dummy_accuracy:.2f}%")
print(f"Acurácia da Árvore de Decisão: {tree_accuracy:.2f}%")
print(f"Acurácia do KNN: {knn_accuracy:.2f}%")
```

## 7. Salvamento do Melhor Modelo

O melhor modelo é salvo usando pickle para uso futuro:

```python
import pickle

# Salvamento do modelo KNN
with open('modelo_knn.pkl', 'wb') as arquivo:
    pickle.dump(knn, arquivo)

# Salvamento do transformador
with open('modelo_transformer.pkl', 'wb') as arquivo:
    pickle.dump(transformer, arquivo)
```

## 8. Uso do Modelo

Carregamento do modelo salvo e previsão para novos dados:

```python
# Carregar o modelo e transformador
modelo_knn = pickle.load(open('modelo_knn.pkl', 'rb'))
modelo_transformer = pickle.load(open('modelo_transformer.pkl', 'rb'))

# Novo cliente
novo_cliente = {
    'idade': [40],
    'estado_civil': ['solteiro (a)'],
    'escolaridade': ['superior'],
    'inadimplente': [0],
    'saldo': [23400],
    'fez_emprestimo': ['nao'],
    'tempo_ult_contato': [10],
    'numero_contatos': [2]
}
novo_cliente_df = pd.DataFrame(novo_cliente)
novo_cliente_transformado = modelo_transformer.transform(novo_cliente_df)

# Previsão
previsao = modelo_knn.predict(novo_cliente_transformado)
print("Previsão para o novo cliente:", le.inverse_transform(previsao))
```

## Conclusão

Este projeto forneceu uma abordagem detalhada para a análise e previsão da aderência de investimentos utilizando técnicas de machine learning. Através da transformação de dados, ajuste e comparação de modelos, e salvamento e uso dos melhores modelos, oferecemos um fluxo de trabalho completo para prever o comportamento dos clientes em relação aos investimentos.
