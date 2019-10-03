# Keras: 2-D Convolutional Neural Network (2-D CNN) para Classifica��o Multiclasse
# Dispon��vel em: https://www.dobitaobyte.com.br/convolutional-neural-network-com-keras/

# Importando pacotes e subpacotes
from keras.datasets import mnist                                                           # Dataset MNIST
from keras.utils import to_categorical                                                     # Categorização em clases
from keras.models import Sequential                                                        # Modelos do Keras (redes neurais)
from keras.layers import Dense, Conv2D, Flatten                                            # Camadas do Keras (redes neurais)
 
# Extraindo os dados 
(X_train, y_train), (X_test, y_test) = mnist.load_data()                                   # Dividindo os dados em entradas e saídas para treinamento e testes

# Formatando os conjuntos de entrada 
X_train = X_train.reshape(60000,28,28,1)                                                   # Entrada de treinamento
X_test = X_test.reshape(10000,28,28,1)                                                     # Entrada de testes
 
# Formatando os conjuntos de sa��da 
y_train = to_categorical(y_train)                                                          # Saída de treinamento
y_test = to_categorical(y_test)                                                            # Saída de testes
 
# Estruturando a rede
model = Sequential()                                                                       # Modelo como pilha linear de camadas
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))             # Entrada multidimensional, primeira camada oculta com 64 neurônios, função de ativação do tipo Rectified Linear Unit (ReLU) e tamanho de kernel (filtro) de 3x3
model.add(Conv2D(32, kernel_size=3, activation='relu'))                                   # Segunda camada oculta com 32 neurônios, função de ativação do tipo Rectified Linear Unit (ReLU) e tamanho de kernel (filtro) de 3x3
model.add(Flatten())                                                                       # Camada conectando a parte convolucional e a parte densa
model.add(Dense(10, activation='softmax'))                                              # Última camada (saída) com 10 neurônios e função de ativação do tipo Softmax 

# Configurando o processo de treinamento                                                   
model.compile(optimizer='adam',                                                            # Algoritmo de otimização Adam
              loss='categorical_crossentropy',                                             # Função de perda do tipo Categorical Cross-Entropy
			  metrics=['accuracy'])                                                        # Métrica: função usada para medir o desempenho do modelo

# Treinando o modelo 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# Realizando a predição sobre o conjunto de testes 
model.predict(X_test[:5])
y_test[:5]