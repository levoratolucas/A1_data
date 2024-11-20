import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


df = pd.read_csv('aula_11_11.csv', sep=';', encoding='latin1')



# _______________________________________ tarefa 6 ____________________________________________________
df['safra'] = pd.to_datetime(df['data_resposta'], format='%d/%m/%Y %H:%M:%S').dt.year
# _____________________________________________________________________________________________________



# _______________________________________ tarefa 1 ____________________________________________________
df['nps'] = df['nota'].apply(lambda x: 'detrator' if x < 6 else ('neutro' if x < 9 else 'promotor'))
# _____________________________________________________________________________________________________




# _______________________________________ tarefa 2 e 3 ________________________________________________
dfBrasil = df[(df['mercado'] == 'BRASIL') & (df['Grupo de Produto'] == 'Grupo 4')]
# _____________________________________________________________________________________________________



# _______________________________________ tarefa 4 ____________________________________________________

contagem_classes = dfBrasil['nps'].value_counts()

total = contagem_classes.sum()


porcentagem = (contagem_classes / total) * 100

print('\nDesbalanceamento nps Brasil Grupo 4')
print(contagem_classes)
print('porcentagem')
print(porcentagem)
# _____________________________________________________________________________________________________

# _______________________________________ tarefa 5 ____________________________________________________
regioes = {
    'PR': 'Sul', 'SC': 'Sul', 'RS': 'Sul',
    'SP': 'Sudeste', 'RJ': 'Sudeste', 'MG': 'Sudeste', 'ES': 'Sudeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'AM': 'Norte', 'RR': 'Norte', 'AP': 'Norte', 'PA': 'Norte', 'TO': 'Norte', 'RO': 'Norte', 'AC': 'Norte',
    'BA': 'Nordeste', 'PE': 'Nordeste', 'CE': 'Nordeste', 'RN': 'Nordeste', 'PB': 'Nordeste',
    'SE': 'Nordeste', 'AL': 'Nordeste', 'PI': 'Nordeste', 'MA': 'Nordeste'
}

df['região'] = df['estado'].map(regioes).fillna('Outro')

# Exibir os dados atualizados
# print(df[['estado', 'região']])
# _____________________________________________________________________________________________________


# _________________________________________ tarefa 7 __________________________________________________
contagem_classes = dfBrasil['nps'].groupby(dfBrasil['safra']).value_counts()

total = contagem_classes.sum()


porcentagem = (contagem_classes / total) * 100

# print(contagem_classes)

# print(porcentagem)

# Criar uma tabela pivô com as contagens
tabela_contagem = dfBrasil.pivot_table(
    index='safra',
    columns='nps',
    aggfunc='size',
    fill_value=0
)

# Calcular as porcentagens para cada safra
tabela_porcentagem = tabela_contagem.div(tabela_contagem.sum(axis=1), axis=0) * 100

# Combinar contagem e porcentagem em uma única tabela
tabela_final = tabela_contagem.astype(str) + " (" + tabela_porcentagem.round(2).astype(str) + "%)"

# Exibir a tabela final
print(tabela_final)

# _____________________________________________________________________________________________________




# _______________________________________ tarefa 8 ____________________________________________________
colunas_grupo_4 = ['capacidade operacional (hectares por hora) (csat)', 
                   'adequação as diversas operações e implementos (csat)', 
                   'facilidade de operação (csat)', 
                   'conforto e ergonomia (csat)', 
                   'disponibilidade e confiabilidade mecânica  (csat)',
                   'facilidade para realização de manutenções (csat)',
                   'custo de manutenção (csat)',
                   'consumo de combustível (litros por hectares) (csat)',
                   'adaptabilidade as mais diversas condições de trabalho (csat)',
                   'facilidade de uso do piloto automático (csat)',
                   'geração e transmissão de dados para gestão da frota (csat)',
                   'geração e transmissão de dados para gestão agrícola (csat)']
# _____________________________________________________________________________________________________


# _______________________________________ tarefa 9 ____________________________________________________

correlacao_spearman = dfBrasil[colunas_grupo_4].corr(method='spearman')

variaveis_numericas = dfBrasil.select_dtypes(include=[np.number]).columns.tolist()

correlacao_nota = dfBrasil[variaveis_numericas].corrwith(dfBrasil['nota'], method='spearman')

correlacao_nota = correlacao_nota.sort_values(ascending=False)

# Exibir o ranqueamento
print("\nRanqueamento das variáveis com base na correlação com 'nota':")
print(correlacao_nota.drop('nota'))

# _____________________________________________________________________________________________________



# _______________________________________ tarefa 10____________________________________________________


dfBrasilCentroOeste = dfBrasil[dfBrasil['região'] == 'Centro-Oeste']

X_detrator = dfBrasilCentroOeste[colunas_grupo_4]

y_detrator = dfBrasilCentroOeste['nps'].map({'detrator': 1, 'neutro': 0, 'promotor': 0})


X_train, X_test, y_train, y_test = train_test_split(X_detrator, y_detrator, test_size=0.25, random_state=42)

model_detrator_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)

model_detrator_XGB.fit(X_train, y_train)


model_detrator_Random = RandomForestClassifier(n_estimators=60, random_state=42)

model_detrator_Random.fit(X_train, y_train)




X_neutro = dfBrasilCentroOeste[colunas_grupo_4]

y_neutro = dfBrasilCentroOeste['nps'].map({'detrator': 0, 'neutro': 1, 'promotor': 0})


X_train, X_test, y_train, y_test = train_test_split(X_neutro, y_neutro, test_size=0.25, random_state=42)

model_neutro_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)

model_neutro_XGB.fit(X_train, y_train)

model_neutro_Random = RandomForestClassifier(n_estimators=60, random_state=42)

model_neutro_Random.fit(X_train, y_train)


# _____________________________________________________________________________________________________


# Importância das variáveis no modelo RandomForest para 'detrator'
importancia_random_detrator = pd.Series(model_detrator_Random.feature_importances_, index=colunas_grupo_4).sort_values(ascending=False)

# Importância das variáveis no modelo XGBoost para 'detrator'
importancia_xgb_detrator = pd.Series(model_detrator_XGB.feature_importances_, index=colunas_grupo_4).sort_values(ascending=False)

# Importância das variáveis no modelo RandomForest para 'neutro'
importancia_random_neutro = pd.Series(model_neutro_Random.feature_importances_, index=colunas_grupo_4).sort_values(ascending=False)

# Importância das variáveis no modelo XGBoost para 'neutro'
importancia_xgb_neutro = pd.Series(model_neutro_XGB.feature_importances_, index=colunas_grupo_4).sort_values(ascending=False)

# Exibir as top 10 variáveis para cada modelo
print("Top 10 variáveis - RandomForest (Detrator):\n", importancia_random_detrator.head(10))
print("\nTop 10 variáveis - XGBoost (Detrator):\n", importancia_xgb_detrator.head(10))
print("\nTop 10 variáveis - RandomForest (Neutro):\n", importancia_random_neutro.head(10))
print("\nTop 10 variáveis - XGBoost (Neutro):\n", importancia_xgb_neutro.head(10))
