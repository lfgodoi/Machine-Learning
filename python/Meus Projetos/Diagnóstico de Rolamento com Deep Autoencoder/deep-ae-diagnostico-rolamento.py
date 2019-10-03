# Diagnóstico com Deep Autoencoder
# -------------------------------------------------
# leonardo Franco de Godói
# 26/09/2019
# -------------------------------------------------

# Importando pacotes
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import keras
from keras.callbacks import EarlyStopping

# Definindo o diretório dos datasets originais
dir_health = 'health_30_2.csv'
dir_inner = 'inner_30_2.csv'
dir_outer = 'outer_30_2.csv'
dir_ball = 'ball_30_2.csv'
dir_comb = 'comb_30_2.csv'

# Número de classes (4 falhas + 1 saudável = 5)
num_classes = 5

# Definindo os parâmetros da rede neural
hidden_layer1 = 256
hidden_layer2 = 128
hidden_layer3 = 64
hidden_layer4 = 32
NUM_EPOCHS = 100
BATCH_SIZE = 1024
act_func = 'relu'

# Carregando os datasets originais (8 sensores x 1048560 medições)
# 1 - Dataset saudável (health)
# 2 - Dataset com falha no anel interno (inner)
# 3 - Dataset com falha no anel externo (outer)
# 4 - Dataset com falha nos elementos rolantes (ball)
# 5 - Dataset com falha combinada dos anéis (comb)
dataset_health = pd.read_csv(dir_health, sep='\t', header=-1)
dataset_inner = pd.read_csv(dir_inner, sep='\t', header=-1)
dataset_outer = pd.read_csv(dir_outer, sep='\t', header=-1)
dataset_ball = pd.read_csv(dir_ball, sep='\t', header=-1)
dataset_comb = pd.read_csv(dir_comb, sep='\t', header=-1)

# Removendo a última coluna (vazia)
dataset_health = dataset_health.iloc[:,:-1] 
dataset_inner = dataset_inner.iloc[:,:-1]
dataset_outer = dataset_outer.iloc[:,:-1]
dataset_ball = dataset_ball.iloc[:,:-1]
dataset_comb = dataset_comb.iloc[:,:-1]

# Definindo rótulos para as colunas e ajustando as linhas
dataset_health.columns = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
                          'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
dataset_inner.columns = dataset_health.columns
dataset_inner.index = dataset_health.index
dataset_outer.columns = dataset_health.columns
dataset_outer.index = dataset_health.index
dataset_ball.columns = dataset_health.columns
dataset_ball.index = dataset_health.index
dataset_comb.columns = dataset_health.columns
dataset_comb.index = dataset_health.index

# Definindo conjuntos de treinamento e validação
train_size = 50000
test_size = 20000
X_train = dataset_health[:train_size].append([dataset_inner[:train_size],
                                          dataset_outer[:train_size],
                                          dataset_ball[:train_size],
                                          dataset_comb[:train_size]])
X_test = dataset_health[:test_size].append([dataset_inner[:test_size],
                                          dataset_outer[:test_size],
                                          dataset_ball[:test_size],
                                          dataset_comb[:test_size]])
y_train = np.repeat(np.array([0,1,2,3,4]), train_size)
y_test = np.repeat(np.array([0,1,2,3,4]), test_size)

# Convertendo vetores de classes em matrizes binárias
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Modelo do Autoencoder
my_input = Input(shape=(X_train.shape[1],))
encoded = Dense(hidden_layer1, activation=act_func)(my_input)
encoded = Dense(hidden_layer2, activation=act_func)(encoded)
encoded = Dense(hidden_layer3, activation=act_func)(encoded)
encoded = Dense(hidden_layer4, activation=act_func)(encoded) # Gargalo
decoded = Dense(hidden_layer3, activation=act_func)(encoded)
decoded = Dense(hidden_layer2, activation=act_func)(decoded)
decoded = Dense(hidden_layer1, activation=act_func)(decoded)
decoded = Dense(X_train.shape[1], activation='sigmoid')(decoded)
autoencoder = Model(my_input, decoded)

# Compilando o modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Criando um mecanismo de Early Stopping baseado na perda
es_loss = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# Treinando o Autoencoder
autoencoder.fit(X_train, X_train,
                epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[es_loss],
                shuffle=True,
                validation_data=(X_test, X_test))

# Inserindo a camada de classificação
output = Dense(num_classes, activation='softmax')(encoded)

# Redefinindo o modelo da nova rede
autoencoder = Model(my_input, output)

# Compilando o novo modelo
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
autoencoder.fit(X_train,
                y_train,
                epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test))

# Avaliando o desempenho da rede
score = autoencoder.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])









