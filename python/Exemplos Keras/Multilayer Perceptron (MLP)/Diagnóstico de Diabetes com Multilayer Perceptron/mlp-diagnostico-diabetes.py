# Keras: Multilayer Perceptron (MLP) para Diagnóstico de Diabetes
# Disponível em: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Importando pacotes e subpacotes
from numpy import loadtxt                                                           # Leitura de dados de um arquivo de texto
from keras.models import Sequential                                                 # Modelos do Keras (redes neurais)
from keras.layers import Dense                                                      # Camadas do Keras (redes neurais)

# Carregando o dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')                       # Extraindo o conteúdo de um arquivo no formato CSV (Comma-Separated Values)

# Dividindo os dados em variáveis de entrada (X) e saída (y)
X = dataset[:,0:8]                                                                  # Todas as linhas; colunas de 0 a 7. Matriz: amostras (n) x variáveis de entrada (8)
y = dataset[:,8]                                                                    # Todas as linhas; coluna 8 (última). Vetor: amostras (n) x variável de saída (1)

# Estruturando a rede
model = Sequential()                                                                # Modelo como pilha linear de camadas
model.add(Dense(12, input_dim=8, activation='relu'))                                # Entrada com 8 neurônios, primeira camada oculta com 12 neurônios e função de ativação do tipo Rectified Linear Unit (ReLU)
model.add(Dense(8, activation='relu'))                                              # Segunda camada oculta com 8 neurônios e função de ativação do tipo Rectified Linear Unit (ReLU)
model.add(Dense(1, activation='sigmoid'))                                           # Última camada (saída) com 1 neurônio e função de ativação do tipo Sigmóide

# Configurando o processo de aprendizado
model.compile(loss='binary_crossentropy',                                           # Função de perda do tipo Binary Cross-Entropy
              optimizer='adam',                                                     # Algoritmo de otimização Adam
			  metrics=['accuracy'])                                                 # Métrica: função usada para medir o desempenho do modelo
			  
# Treinando o modelo
model.fit(X, y, epochs=150, batch_size=10)                                          # Conjunto de dados (entradas e saídas) de treinamento
_, accuracy = model.evaluate(X, y)                                                  # Valores de perda e métrica do modelo
print('Accuracy: %.2f' % (accuracy*100))                                            # Mostrando a precisão obtida