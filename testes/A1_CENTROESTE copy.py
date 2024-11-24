import unicodedata
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

csat_columns = [col for col in df.columns if 'csat' in col.lower()]
print(df.columns)

for col in csat_columns:
    if col in df.columns:
        df[col] = df[col].fillna(-1)

# df[csat_columns] = df[csat_columns].fillna(-1)





df_filtered = df[csat_columns + ['região', 'safra', 'nps','grupo_de_produto', 'mercado','nota']]

print(df_filtered['nota'])


spanish_patterns = ['like/dislike', 'multiple', 'input']
spanish_csat_columns = [col for col in csat_columns if any(pattern in col for pattern in spanish_patterns)]
df_filtered = df_filtered.drop(columns=spanish_csat_columns)
print(df_filtered.head())
df_filtered_brazil_group = df_filtered[(df['mercado'].str.lower().str.strip() == 'brasil') &
                                       (df['grupo_de_produto'].str.lower().str.strip() == 'grupo 4')]

# Calculando a porcentagem de valores não nulos nas colunas de CSAT
# Calculando a porcentagem de valores maiores que 0 nas colunas de CSAT
csat_greater_than_zero_percentage = (df_filtered_brazil_group[csat_columns] > 0).mean() * 100

# Exibir as colunas que atendem à condição
# print(csat_columns_30plus)# Ajuste de filtro para 30% a 60%
csat_columns_filtered = csat_greater_than_zero_percentage[(csat_greater_than_zero_percentage > 20)].index



correlacao_nota = df_filtered_brazil_group[csat_columns_filtered].corrwith(df_filtered_brazil_group['nota'], method='spearman')

correlacao_nota = correlacao_nota.sort_values(ascending=False)

# Exibir o ranqueamento
print("\nRanqueamento das variáveis com base na correlação com 'nota':")
print(correlacao_nota.head(10))


X_detrator = df_filtered_brazil_group[csat_columns_filtered]

y_detrator = df_filtered_brazil_group['nps'].map({'detrator': 1, 'neutro': 1, 'promotor': 0})


X_train, X_test, y_train, y_test = train_test_split(X_detrator, y_detrator, test_size=0.25, random_state=42)

model_detrator_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)

model_detrator_XGB.fit(X_train, y_train)

importancia_xgb_detrator = pd.Series(model_detrator_XGB.feature_importances_, index=csat_columns_filtered).sort_values(ascending=False)

print(importancia_xgb_detrator.head(10))
