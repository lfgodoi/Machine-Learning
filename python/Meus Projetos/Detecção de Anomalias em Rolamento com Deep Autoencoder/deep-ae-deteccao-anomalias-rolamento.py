# Detecção de anomalia com Deep Autoencoder
# -------------------------------------------------
# leonardo Franco de Godói
# 18/09/2019
# -------------------------------------------------

# Importando pacotes
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow import set_random_seed
from sklearn import preprocessing
import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt

# Definindo o diretório dos datasets originais
dir_health = 'health_30_2.csv'
dir_fault = 'outer_30_2.csv'

# Definindo os parâmetros da rede neural
hidden_layer1 = 12
hidden_layer2 = 6
hidden_layer3 = 3
NUM_EPOCHS = 100
BATCH_SIZE = 1000
act_func = 'selu'

# Carregando os datasets originais
# 1 - Dataset saudável: 8 sensores x 1048560 medições
# 2 - Dataset com anomalia: 8 sensores x 1048560 medições
dataset_health = pd.read_csv(dir_health, sep='\t', header=-1)
dataset_fault = pd.read_csv(dir_fault, sep='\t', header=-1)

# Estruturando dois novos datasets
# 1 - Dataset de treinamento: 8 sensores x 600000 medições (todos saudáveis)
# 2 - Dataset de validação: 8 sensores x 600000 medições (50% com anomalia)
dataset_train = dataset_health[:600000]
dataset_test = dataset_fault[600001:900001]
dataset_test = dataset_test.append(dataset_fault[:300000]) 

# Removendo a última coluna (vazia)
dataset_train = dataset_train.iloc[:, :-1]
dataset_test = dataset_test.iloc[:, :-1]

# Definindo rótulos para as colunas e ajustando as linhas
dataset_train.columns = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
                          'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
dataset_test.columns = dataset_train.columns
dataset_test.index = dataset_train.index

# Criando novos datasets com dados normalizados (0 a 1)
scaler = preprocessing.MinMaxScaler()
data_train = pd.DataFrame(scaler.fit_transform(dataset_train), 
                              columns=dataset_train.columns, 
                              index=dataset_train.index)
data_test = pd.DataFrame(scaler.fit_transform(dataset_test), 
                              columns=dataset_test.columns, 
                              index=dataset_test.index)

# Definindo a semente (seed) aleatoriamente
seed(10)
set_random_seed(10)

# Definindo o modelo do Deep Autoencoder
model = Sequential()
model.add(Dense(hidden_layer1, activation=act_func, 
                input_shape=(data_train.shape[1],)))
model.add(Dense(hidden_layer2, activation=act_func))
model.add(Dense(hidden_layer3, activation=act_func))
model.add(Dense(hidden_layer2, activation=act_func))
model.add(Dense(hidden_layer1, activation=act_func)) 
model.add(Dense(data_train.shape[1]))

# Mostrando a estrutura do modelo
model.summary()

# Compilando o modelo para treinamento
# Otimizador: Adam (Adaptive Moment Estimation)
# Função de perda: MSE (Mean Square Error)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Criando o mecanismo de Early Stopping para evitar overfitting
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# Treinando o modelo
# 5% dos dados de treinamento são reservados para validação
# É declarado um callback para o Early Stopping
history_train = model.fit(np.array(data_train), np.array(data_train),
                   batch_size=BATCH_SIZE,
                   epochs=NUM_EPOCHS,
                   validation_split=0.05,
                   callbacks=[es],
                   verbose=1)

# Gerando predições a partir dos dados de treinamento
pred_train = model.predict(np.array(data_train))
pred_train = pd.DataFrame(pred_train, columns=data_train.columns)
pred_train.index = data_train.index

# Calculando o MAE das predições sobre os dados de treinamento
# Média móvel com janela de 10000 amostras é usada para suavizar a curvas
scored_train = pd.DataFrame(index=data_train.index)
scored_train['Loss MAE'] = np.mean(np.abs(pred_train-data_train), axis = 1)
scored_train = scored_train.rolling(window=10000).mean()

# Gerando predições a partir dos dados de validação
pred_test = model.predict(np.array(data_test))
pred_test = pd.DataFrame(pred_test, columns=data_test.columns)
pred_test.index = data_test.index

# Calculando o MAE das predições sobre os dados de validação
# Média móvel com janela de 10000 amostras é usada para suavizar a curvas
scored_test = pd.DataFrame(index=data_test.index)
scored_test['Loss MAE'] = np.mean(np.abs(pred_test-data_train), axis = 1)
scored_test = scored_test.rolling(window=10000).mean()

# Plotando o desempenho em função do MAE
plt.figure(1)
plt.subplot(211)
plt.plot(scored_train, label='Operando normalmente', color='green')
plt.xlabel('Medição')
plt.ylabel('MAE')
plt.legend()
plt.subplot(212)
plt.plot(scored_test[:300000], label='Operando normalmente', color='green')
plt.plot(scored_test[300001:600000], 
         label='Operando com falha', color='red')
plt.xlabel('Medição')
plt.ylabel('MAE')
plt.legend()





