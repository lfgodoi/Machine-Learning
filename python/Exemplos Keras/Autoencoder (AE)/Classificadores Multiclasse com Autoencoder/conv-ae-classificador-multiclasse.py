# Keras: Convolutional Autoencoder (CAE) para Classificação Multiclasse
# Disponível em: https://blog.keras.io/building-autoencoders-in-keras.html

# Importando pacotes e subpacotes
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Definindo a entrada
input_img = Input(shape=(28,28,1))

# Encoder como camadas de convolução
encoded = Conv2D(16,(3, 3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)

# Decoder como camadas de deconvolução
decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(16, (3, 3), activation='relu')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(1,(3,3), activation='sigmoid', padding='same')(decoded)

# Criação do modelo
autoencoder = Model(input_img, decoded)

# Compilação do modelo
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Carregando o dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalizando os dados de 0 a 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# Achatando o conjunto
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Treinando o modelo
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Obtendo a imagem reconstruída
decoded_imgs = autoencoder.predict(x_test)

# Representando os resultados graficamente
plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()














