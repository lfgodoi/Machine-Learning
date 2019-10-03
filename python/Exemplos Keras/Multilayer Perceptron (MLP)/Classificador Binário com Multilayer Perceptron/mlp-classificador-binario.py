# Keras: Multilayer Perceptron (MLP) para Classificação Binária
# Disponível em: https://keras.io/getting-started/sequential-model-guide/

# Importando pacotes e subpacotes
import numpy as np                                           # NumPy (para arrays multidimensionais)
from keras.models import Sequential                          # Modelos do Keras (redes neurais)
from keras.layers import Dense, Dropout                      # Camadas do Keras (redes neurais)

# Gerando dados aleatórios
x_train = np.random.random((1000, 20))                       # Matriz de entrada para treinamento: 1000 (amostras) x 20 (entradas)
y_train = np.random.randint(2, size=(1000, 1))               # Matriz de saída para treinamento com valores binários: 1000 (amostras) x 1 (saída)           
x_test = np.random.random((100, 20))                         # Matriz de entrada para testes: 100 (amostras) x 20 (entradas)
y_test = np.random.randint(2, size=(100, 1))                 # Matriz de saída para testes com valores binários: 100 (amostras) x 1 (saída) 

# Estruturando a rede
model = Sequential()                                         # Modelo como pilha linear de camadas
model.add(Dense(64, input_dim=20, activation='relu'))        # Entrada com 20 neurônios, primeira camada oculta 64 neurônios e função de ativação do tipo Rectified Linear Unit (ReLU)
model.add(Dropout(0.5))                                      # Taxa de dropout
model.add(Dense(64, activation='relu'))                      # Segunda camada oculta com 64 neurônios e função de ativação do tipo Rectified Linear Unit (ReLU)
model.add(Dropout(0.5))                                      # Taxa de dropout
model.add(Dense(1, activation='sigmoid'))                    # Última camada (saída) com 1 neurônio e função de ativação do tipo Sigmóide

# Configurando o processo de aprendizado
model.compile(loss='binary_crossentropy',                    # Função de perda do tipo Binary Cross-Entropy
              optimizer='rmsprop',                           # Algoritmo de otimização RMSprop
              metrics=['accuracy'])                          # Métrica: função usada para medir o desempenho do modelo

# Treinando o modelo
model.fit(x_train, y_train,                                  # Conjunto de dados (entradas e saídas) de treinamento
          epochs=20,                                         # Número de épocas
          batch_size=128)                                    # Tamanho de batch

# Avaliando o desempenho
_, accuracy = model.evaluate(x_test, y_test, batch_size=128)       # Valores da perda e métrica do modelo
print('Accuracy: %.2f' % (accuracy*100), '%') 