# TensorFlow: Autoencoder (AE) Modular
# Disponí­vel em: https://rubikscode.net/2018/11/26/3-ways-to-implement-autoencoders-with-tensorflow-and-python/

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from autoencoder_tensorflow import Autoencoder
import matplotlib.pyplot as plt

# Importando os dados
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Preparação dos dados
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Implementação
autoencodertf = Autoencoder(x_train.shape[1], 32)
autoencodertf.train(x_train, x_test, 100, 100)
encoded_img = autoencodertf.getEncodedImage(x_test[1])
decoded_img = autoencodertf.getDecodedImage(x_test[1])

# Resultados
plt.figure(figsize=(20, 4))
subplot = plt.subplot(2, 10, 1)
plt.imshow(x_test[1].reshape(28, 28))
plt.gray()
subplot.get_xaxis().set_visible(False)
subplot.get_yaxis().set_visible(False)

subplot = plt.subplot(2, 10, 2)
plt.imshow(decoded_img.reshape(28, 28))
plt.gray()
subplot.get_xaxis().set_visible(False)
subplot.get_yaxis().set_visible(False)