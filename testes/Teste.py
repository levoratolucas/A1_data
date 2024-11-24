import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
grupo = 'Grupo 6'

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


# _____________________________________________________________________________________________________



# _______________________________________ tarefa 2 e 3 ________________________________________________
dfBrasil = df[(df['mercado'] == 'BRASIL') & (df['Grupo de Produto'] == grupo)]
# _____________________________________________________________________________________________________



# _______________________________________ tarefa 4 ____________________________________________________

contagem_classes = dfBrasil['nps'].value_counts()

total = contagem_classes.sum()


contagem_classes = dfBrasil.groupby(['nps']).size().reset_index(name='Contagem')

# Salvar o DataFrame em um arquivo CSV com delimitador ';' e codificação UTF-8
contagem_classes.to_csv('nps_brasil_{}.csv'.format(grupo), sep=';', encoding='utf-8', index=False)

# Exibir mensagem de sucesso
print("desbalanceamento do grupo salvo com sucesso")


contagem_classes = dfBrasil.groupby(['região', 'nps']).size().reset_index(name='Contagem')

# Salvar o DataFrame em um arquivo CSV com delimitador ';' e codificação UTF-8
contagem_classes.to_csv('nps_regiao_{}.csv'.format(grupo), sep=';', encoding='utf-8', index=False)

# Exibir mensagem de sucesso
print("desbalanceamento do grupo salvo com sucesso por região")

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
    fill_value=0)

# Calcular as porcentagens para cada safra
tabela_porcentagem = tabela_contagem.div(tabela_contagem.sum(axis=1), axis=0) * 100

# Combinar contagem e porcentagem em uma única tabela
tabela_final = tabela_contagem.astype(str) + " (" + tabela_porcentagem.round(2).astype(str) + "%)"

# Calcular o total de contagens para todas as colunas
totais_contagem = tabela_contagem.sum(axis=0)
totais_porcentagem = (totais_contagem / totais_contagem.sum()) * 100

# Criar a linha de total
linha_total = totais_contagem.astype(str) + " (" + totais_porcentagem.round(2).astype(str) + "%)"
linha_total.name = "total"

# Usar pd.concat para adicionar a linha de total
tabela_final = pd.concat([tabela_final, linha_total.to_frame().T])

# Salvar a tabela final como CSV
tabela_final.to_csv("safra{}.csv".format(grupo), index=True)

# Exibir a tabela final (opcional)
# print(tabela_final)

# # _____________________________________________________________________________________________________




# # _______________________________________ tarefa 8 ____________________________________________________
# colunas_grupo = ['capacidade operacional (hectares por hora) (csat)', 
#                    'adequação as diversas operações e implementos (csat)', 
#                    'facilidade de operação (csat)', 
#                    'conforto e ergonomia (csat)', 
#                    'disponibilidade e confiabilidade mecânica  (csat)',
#                    'facilidade para realização de manutenções (csat)',
#                    'custo de manutenção (csat)',
#                    'consumo de combustível (litros por hectares) (csat)',
#                    'adaptabilidade as mais diversas condições de trabalho (csat)',
#                    'facilidade de uso do piloto automático (csat)',
#                    'geração e transmissão de dados para gestão da frota (csat)',
#                    'geração e transmissão de dados para gestão agrícola (csat)']

# 1. Selecionar colunas numéricas
colunas_numericas = df.select_dtypes(include=[np.number])

# 2. Filtrar colunas que contenham 'csat' no nome
colunas_csat = colunas_numericas.filter(like='csat')

# 3. Filtrar colunas com pelo menos 50% de valores preenchidos
colunas_relevantes = [col for col in colunas_csat if df[col].notnull().mean() >= 0.3]

# Exibir as colunas selecionadas
print("Colunas selecionadas:", colunas_relevantes)

colunas_grupo = colunas_relevantes

# # _____________________________________________________________________________________________________


# # _______________________________________ tarefa 9 ____________________________________________________

correlacao_spearman = dfBrasil[colunas_grupo].corr(method='spearman')

variaveis_numericas = dfBrasil.select_dtypes(include=[np.number]).columns.tolist()

correlacao_nota = dfBrasil[variaveis_numericas].corrwith(dfBrasil['nota'], method='spearman')

correlacao_nota = correlacao_nota.sort_values(ascending=False)

# Transformar a Series em um DataFrame
correlacao_nota_df = correlacao_nota.drop('nota').head(10).reset_index()
correlacao_nota_df.columns = ['Variável', 'Correlação']

# Salvar o DataFrame em um arquivo CSV
correlacao_nota_df.to_csv(f'spearman_{grupo}.csv', sep=';', encoding='utf-8', index=False)

# Exibir mensagem de sucesso
print("Correlação Spearman salva com sucesso!")

# # # _____________________________________________________________________________________________________


def calcular_metricas_e_roc(model, X_test, y_test, label):
    # Previsões e probabilidades
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Exibir as métricas
    print(f"{label} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label}")
    plt.legend(loc="best")
    plt.show()

# _______________________________________ tarefa 10____________________________________________________
# Lista de todas as regiões possíveis
regioes_unicas = df['região'].unique()

# Dicionários para armazenar os modelos para cada target
modelos_detrator_XGB = {}
modelos_detrator_Random = {}

modelos_neutro_XGB = {}
modelos_neutro_Random = {}

modelos_promotor_XGB = {}
modelos_promotor_Random = {}

# Loop pelas regiões
for regiao in regioes_unicas:
    # Filtrar os dados pela região
    df_regiao = df[df['região'] == regiao]

    # Detratores
    X_detrator = df_regiao[colunas_grupo]
    y_detrator = df_regiao['nps'].map({'detrator': 1, 'neutro': 0, 'promotor': 0})

    X_train, X_test, y_train, y_test = train_test_split(X_detrator, y_detrator, test_size=0.25, random_state=42)

    # Modelo XGBoost para detratores
    model_detrator_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)
    model_detrator_XGB.fit(X_train, y_train)
    modelos_detrator_XGB[regiao] = model_detrator_XGB

    # Modelo Random Forest para detratores
    model_detrator_Random = RandomForestClassifier(n_estimators=60, random_state=42)
    model_detrator_Random.fit(X_train, y_train)
    modelos_detrator_Random[regiao] = model_detrator_Random

    # Neutros
    X_neutro = df_regiao[colunas_grupo]
    y_neutro = df_regiao['nps'].map({'detrator': 0, 'neutro': 1, 'promotor': 0})

    X_train, X_test, y_train, y_test = train_test_split(X_neutro, y_neutro, test_size=0.25, random_state=42)

    # Modelo XGBoost para neutros
    model_neutro_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)
    model_neutro_XGB.fit(X_train, y_train)
    modelos_neutro_XGB[regiao] = model_neutro_XGB

    # Modelo Random Forest para neutros
    model_neutro_Random = RandomForestClassifier(n_estimators=60, random_state=42)
    model_neutro_Random.fit(X_train, y_train)
    modelos_neutro_Random[regiao] = model_neutro_Random

    # Promotores
    X_promotor = df_regiao[colunas_grupo]
    y_promotor = df_regiao['nps'].map({'detrator': 1, 'neutro': 1, 'promotor': 0})

    X_train, X_test, y_train, y_test = train_test_split(X_promotor, y_promotor, test_size=0.25, random_state=42)

    # Modelo XGBoost para promotores
    model_promotor_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)
    model_promotor_XGB.fit(X_train, y_train)
    modelos_promotor_XGB[regiao] = model_promotor_XGB

    # Modelo Random Forest para promotores
    model_promotor_Random = RandomForestClassifier(n_estimators=60, random_state=42)
    model_promotor_Random.fit(X_train, y_train)
    modelos_promotor_Random[regiao] = model_promotor_Random
    
    # Função para calcular métricas e exibir a curva ROC

# Loop por todas as regiões e todos os modelos
for regiao in regioes_unicas:
    print(f"\nMétricas para a região: {regiao}")
    
    # Detratores
    model_detrator_XGB = modelos_detrator_XGB[regiao]
    model_detrator_Random = modelos_detrator_Random[regiao]
    print("\nDetratores (XGBoost):")
    # calcular_metricas_e_roc(model_detrator_XGB, X_test, y_test, f"Detratores - {regiao} (XGBoost)")
    print("\nDetratores (Random Forest):")
    # calcular_metricas_e_roc(model_detrator_Random, X_test, y_test, f"Detratores - {regiao} (Random Forest)")
    
    # Neutros
    model_neutro_XGB = modelos_neutro_XGB[regiao]
    model_neutro_Random = modelos_neutro_Random[regiao]
    print("\nNeutros (XGBoost):")
    # calcular_metricas_e_roc(model_neutro_XGB, X_test, y_test, f"Neutros - {regiao} (XGBoost)")
    print("\nNeutros (Random Forest):")
    # calcular_metricas_e_roc(model_neutro_Random, X_test, y_test, f"Neutros - {regiao} (Random Forest)")
    
    # Promotores
    model_promotor_XGB = modelos_promotor_XGB[regiao]
    model_promotor_Random = modelos_promotor_Random[regiao]
    print("\nPromotores (XGBoost):")
    # calcular_metricas_e_roc(model_promotor_XGB, X_test, y_test, f"Promotores - {regiao} (XGBoost)")
    print("\nPromotores (Random Forest):")
    # calcular_metricas_e_roc(model_promotor_Random, X_test, y_test, f"Promotores - {regiao} (Random Forest)")



# Para o Brasil como um todo (total)
dfBrasil = df[df['mercado'] == 'BRASIL']

# Detratores Brasil
X_detrator = dfBrasil[colunas_grupo]
y_detrator = dfBrasil['nps'].map({'detrator': 1, 'neutro': 0, 'promotor': 0})

X_train, X_test, y_train, y_test = train_test_split(X_detrator, y_detrator, test_size=0.25, random_state=42)

# Modelo XGBoost para detratores
model_detrator_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)
model_detrator_XGB.fit(X_train, y_train)
modelos_detrator_XGB['BRASIL'] = model_detrator_XGB

# Modelo Random Forest para detratores
model_detrator_Random = RandomForestClassifier(n_estimators=60, random_state=42)
model_detrator_Random.fit(X_train, y_train)
modelos_detrator_Random['BRASIL'] = model_detrator_Random

# Neutros Brasil
X_neutro = dfBrasil[colunas_grupo]
y_neutro = dfBrasil['nps'].map({'detrator': 0, 'neutro': 1, 'promotor': 0})

X_train, X_test, y_train, y_test = train_test_split(X_neutro, y_neutro, test_size=0.25, random_state=42)

# Modelo XGBoost para neutros
model_neutro_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)
model_neutro_XGB.fit(X_train, y_train)
modelos_neutro_XGB['BRASIL'] = model_neutro_XGB

# Modelo Random Forest para neutros
model_neutro_Random = RandomForestClassifier(n_estimators=60, random_state=42)
model_neutro_Random.fit(X_train, y_train)
modelos_neutro_Random['BRASIL'] = model_neutro_Random

# Promotores Brasil
X_promotor = dfBrasil[colunas_grupo]
y_promotor = dfBrasil['nps'].map({'detrator': 1, 'neutro': 1, 'promotor': 0})

X_train, X_test, y_train, y_test = train_test_split(X_promotor, y_promotor, test_size=0.25, random_state=42)

# Modelo XGBoost para promotores
model_promotor_XGB = XGBClassifier(n_estimators=60, verbosity=3, random_state=42)
model_promotor_XGB.fit(X_train, y_train)
modelos_promotor_XGB['BRASIL'] = model_promotor_XGB

# Modelo Random Forest para promotores
model_promotor_Random = RandomForestClassifier(n_estimators=60, random_state=42)
model_promotor_Random.fit(X_train, y_train)
modelos_promotor_Random['BRASIL'] = model_promotor_Random

# Função para calcular métricas e exibir a curva ROC

regiao = 'BRASIL'

print(f"\nMétricas para a região: {regiao}")
    
    # Detratores
model_detrator_XGB = modelos_detrator_XGB[regiao]
model_detrator_Random = modelos_detrator_Random[regiao]
print("\nDetratores (XGBoost):")
# calcular_metricas_e_roc(model_detrator_XGB, X_test, y_test, f"Detratores - {regiao} (XGBoost)")
print("\nDetratores (Random Forest):")
# calcular_metricas_e_roc(model_detrator_Random, X_test, y_test, f"Detratores - {regiao} (Random Forest)")

# Neutros
model_neutro_XGB = modelos_neutro_XGB[regiao]
model_neutro_Random = modelos_neutro_Random[regiao]
print("\nNeutros (XGBoost):")
# calcular_metricas_e_roc(model_neutro_XGB, X_test, y_test, f"Neutros - {regiao} (XGBoost)")
print("\nNeutros (Random Forest):")
# calcular_metricas_e_roc(model_neutro_Random, X_test, y_test, f"Neutros - {regiao} (Random Forest)")

# Promotores
model_promotor_XGB = modelos_promotor_XGB[regiao]
model_promotor_Random = modelos_promotor_Random[regiao]
print("\nPromotores (XGBoost):")
# calcular_metricas_e_roc(model_promotor_XGB, X_test, y_test, f"Promotores - {regiao} (XGBoost)")
print("\nPromotores (Random Forest):")
# calcular_metricas_e_roc(model_promotor_Random, X_test, y_test, f"Promotores - {regiao} (Random Forest)")



import pandas as pd

# Função para extrair as top 10 variáveis de um modelo e adicionar o nome do modelo
def extrair_top_10_importancias(modelo, colunas, nome_modelo):
    if hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
    elif hasattr(modelo, 'get_booster'):  # Caso seja XGBoost
        importancias = modelo.feature_importances_
    else:
        raise ValueError("Modelo não tem atributo 'feature_importances_'")

    # Criar um DataFrame com as importâncias das variáveis
    df_importancias = pd.DataFrame({
        'modelo': [nome_modelo] * len(colunas),
        'variavel': colunas,
        'importancia': importancias
    })

    # Ordenar pelo valor das importâncias e pegar as 10 mais importantes
    df_importancias = df_importancias.sort_values(by='importancia', ascending=False)
    
    return df_importancias.head(10)


# Armazenando as top 10 variáveis de cada modelo para todas as regiões e para o Brasil
top_10_xgb = []
top_10_random = []

# Percorrer as regiões e o Brasil para coletar as importâncias
for regiao in regioes_unicas:
    # Para XGBoost
    if regiao in modelos_detrator_XGB:
        top_10_xgb.append(extrair_top_10_importancias(modelos_detrator_XGB[regiao], colunas_grupo, f'XGBoost_Detrator_{regiao}'))
        top_10_xgb.append(extrair_top_10_importancias(modelos_neutro_XGB[regiao], colunas_grupo, f'XGBoost_Neutro_{regiao}'))
        top_10_xgb.append(extrair_top_10_importancias(modelos_promotor_XGB[regiao], colunas_grupo, f'XGBoost_Promotor_{regiao}'))
    
    # Para Random Forest
    if regiao in modelos_detrator_Random:
        top_10_random.append(extrair_top_10_importancias(modelos_detrator_Random[regiao], colunas_grupo, f'Random_Detrator_{regiao}'))
        top_10_random.append(extrair_top_10_importancias(modelos_neutro_Random[regiao], colunas_grupo, f'Random_Neutro_{regiao}'))
        top_10_random.append(extrair_top_10_importancias(modelos_promotor_Random[regiao], colunas_grupo, f'Random_Promotor_{regiao}'))

# Para o Brasil (total)
top_10_xgb.append(extrair_top_10_importancias(modelos_detrator_XGB['BRASIL'], colunas_grupo, 'XGBoost_Detrator_BRASIL'))
top_10_xgb.append(extrair_top_10_importancias(modelos_neutro_XGB['BRASIL'], colunas_grupo, 'XGBoost_Neutro_BRASIL'))
top_10_xgb.append(extrair_top_10_importancias(modelos_promotor_XGB['BRASIL'], colunas_grupo, 'XGBoost_Promotor_BRASIL'))

top_10_random.append(extrair_top_10_importancias(modelos_detrator_Random['BRASIL'], colunas_grupo, 'Random_Detrator_BRASIL'))
top_10_random.append(extrair_top_10_importancias(modelos_neutro_Random['BRASIL'], colunas_grupo, 'Random_Neutro_BRASIL'))
top_10_random.append(extrair_top_10_importancias(modelos_promotor_Random['BRASIL'], colunas_grupo, 'Random_Promotor_BRASIL'))

# Concatenando os resultados em DataFrames
df_top_10_xgb = pd.concat(top_10_xgb)
df_top_10_random = pd.concat(top_10_random)

# Salvando os resultados em CSV
df_top_10_xgb.to_csv('XGBOOST_{}.csv'.format(grupo), sep=';', encoding='utf-8', index=False)
df_top_10_random.to_csv('RANDOM_{}.csv'.format(grupo), sep=';', encoding='utf-8', index=False)

print("CSV para XGBoost e Random Forest gerados com as top 10 variáveis.")

