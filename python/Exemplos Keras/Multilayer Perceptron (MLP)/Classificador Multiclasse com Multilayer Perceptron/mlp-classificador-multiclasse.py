# Keras: Multilayer Perceptron (MLP) para classificação multiclasse
# Disponível em: https://keras.io/getting-started/sequential-model-guide/

# Importando pacotes e subpacotes
import keras                                                                                 # Keras (redes neurais)
from keras.models import Sequential                                                          # Modelos do Keras (redes neurais)
from keras.layers import Dense, Dropout                                                      # Camadas do Keras (redes neurais)
from keras.optimizers import SGD                                                             # Otimizador SGD (Stochastic Gradient Descent)

# Gerando dados aleatórios   
import numpy as np                                                                           # NumPy (para arrays multidimensionais)                                                           
x_train = np.random.random((1000, 20))                                                       # Matriz de entrada para treinamento: 1000 (amostras) x 20 (entradas)
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)  # Matriz de saída para treinamento com valores categorizados em 10 classes: 100 (amostras) x 1 (saída) 
x_test = np.random.random((100, 20))                                                         # Matriz de entrada para testes: 100 (amostras) x 20 (entradas)
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)    # Matriz de saída para testes com valores categorizados em 10 classes: 100 (amostras) x 1 (saída) 

# Estruturando a rede
model = Sequential()                                                                         # Modelo como pilha linear de camadas
model.add(Dense(64, input_dim=20, activation='relu'))                                        # Entrada com 20 neurônios, primeira camada oculta com 64 neurônios e função de ativação do tipo Rectified Linear Unit (ReLU)
model.add(Dropout(0.5))                                                                      # Taxa de dropout
model.add(Dense(64, activation='relu'))                                                      # Segunda camada oculta com 64 neurônios e função de ativação do tipo Rectified Linear Unit (ReLU)
model.add(Dropout(0.5))                                                                      # Taxa de dropout
model.add(Dense(10, activation='softmax'))                                                   # Última camada (saída) com 10 neurônios e função de ativação do tipo Softmax

# Configurando o processo de aprendizado
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)                                  # Definindo os parâmetros do otimizador
model.compile(loss='categorical_crossentropy',                                               # Função de perda do tipo Categorical Cross-Entropy
              optimizer=sgd,                                                                 # Algoritmo de otimização SGD
              metrics=['accuracy'])                                                          # Métrica: função usada para medir o desempenho do modelo

# Treinando o modelo
model.fit(x_train, y_train,                                                                  # Conjunto de dados (entradas e saídas) de treinamento
          epochs=20,                                                                         # Número de épocas
          batch_size=128)

# Avaliando o modelo                                                                    # Tamanho de batch
_, accuracy = model.evaluate(x_test, y_test, batch_size=128)  
print('Accuracy: %.2f' % (accuracy*100), '%')                                      # Valores da perda e métrica do modelo