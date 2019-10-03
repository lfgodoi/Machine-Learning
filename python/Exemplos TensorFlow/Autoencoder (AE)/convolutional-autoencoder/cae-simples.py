# TensorFlow: Convolutional Autoencoder (CAE) para Classificação Binária
# Disponível em: http://sabinemaennel.ch/udacity-deeplearning/Convolutional_Autoencoder.html

# Importando pacotes e módulos
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Carregando o dataset MNIST
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

# Manipulando os dados
img = mnist.train.images[2]
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
mnist.train.images.shape[1]
image_size = mnist.train.images.reshape((mnist.train.images.shape[0], 28, 28)).shape[1:]
print('image size: ', image_size)
image_size = mnist.train.images.shape[1]
print('image size: ', image_size)

# Definindo o learning rate
learning_rate = 0.001

# Criando o placeholder para as entradas
inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
print("This is our placeholder for the inputs: ", inputs_)

# Criando o placeholder para as saídas
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')
print("This is our placeholder for the targets: ", targets_)

### Encoder

# Exemplo de camada convolucional no TensorFlow:
# conv1 = tf.layers.conv2d(
#         inputs=input_layer,
#         filters=32,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu)
conv1 = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
print("Convolution Layer conv1", conv1)
print('Now 28x28x16')

# Camada de max pooling
# max_pooling2d(
#    inputs,
#    pool_size,
#    strides,
#    padding='valid',
#    data_format='channels_last',
#    name=None)
maxpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding='same')
print("Maxpooling Layer maxpoo11", maxpool1)
print('Now 14x14x16')

conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=3, padding='same', activation=tf.nn.relu)
print("Convolutinal Layer conv2", conv2)
print('Now 14x14x8')

maxpool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding='same')
print("Maxpooling Layer maxpoo12", maxpool2)
print('Now 7x7x8')

conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=3, padding='same', activation=tf.nn.relu)
print("Convolutinal Layer conv3", conv3)
print('Now 7x7x8')

encoded = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding='same')
print("Encoded", encoded)
print('Now 4x4x8')

### Decoder

#resize_images(
#    images,
#    size,
#    method=ResizeMethod.BILINEAR,
#    align_corners=False
#)
upsample1 = tf.image.resize_images(images=encoded, size=(7,7))
print("Resize Layer  upsample1", upsample1)
print('Now 7x7x8')

conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=3, padding='same', activation=tf.nn.relu)
print("Convolutinal Layer conv4", conv4)
print('Now 7x7x8')

upsample2 = tf.image.resize_images(images=conv4, size=(14, 14))
print("Resize Layer upsample2", upsample2)
print('Now 14x14x8')

conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=3, padding='same', activation=tf.nn.relu)
print("Convolutinal Layer conv5", conv5)
print('Now 14x14x8')

upsample3 = tf.image.resize_images(images=conv5, size=(28, 28))
print("Resize Layer upsample3", upsample3)
print('Now 28x28x8')

conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
print("Convolutinal Layer conv6", conv6)
print('Now 28x28x16')

logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=3, padding='same')
print("Logits", logits)
print('Now 28x28x1')

# Aplicando a função logits sobre a sigmoid para obter a imagem reconstruída
decoded = tf.nn.sigmoid(logits)
print("Decoded", decoded)

# Aplicando a função logits sobre a sigmoid e calculando a perda em cross-entropy
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets_)
print("Loss", loss)

# Obtendo o custo e definindo o otimizador
cost = tf.reduce_mean(loss)
print("Cost", cost)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
print("Opt", opt)

# Iniciando a sessão
sess = tf.Session()

epochs = 20
batch_size = 200
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs = batch[0].reshape((-1, 28, 28, 1))
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                         targets_: imgs})

    print("Epoch: {}/{}...".format(e+1, epochs),
          "Training loss: {:.4f}".format(batch_cost))
    
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[:10]
reconstructed = sess.run(decoded, feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1))})

for images, row in zip([in_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


fig.tight_layout(pad=0.1)

learning_rate = 0.001
inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Agora 28x28x32
maxpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding='same')
# Agora 14x14x32
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Agora 14x14x32
maxpool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding='same')
# Agora 7x7x32
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Agora 7x7x16
encoded = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding='same')
# Agora 4x4x16

### Decoder
upsample1 = tf.image.resize_images(images=encoded, size=(7,7))
# Agora 7x7x16
conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Agora 7x7x16
upsample2 = tf.image.resize_images(images=conv4, size=(14,14))
# Agora 14x14x16
conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Agora 14x14x32
upsample3 = tf.image.resize_images(images=conv5, size=(28,28))
# Agora 28x28x32
conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Agora 28x28x32

logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=3, padding='same')
# Agora 28x28x1

# Aplicando a função logits sobre a sigmoid para obter a imagem reconstruída
decoded = tf.nn.sigmoid(logits)

# Aplicando a função logits sobre a sigmoid e calculando a perda em cross-entropy
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets_)

# Obtendo o custo e definindo o otimizador
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()

epochs = 100
batch_size = 200
# Definindo a quantidade de ruído que está sendo adicionado às imagens
noise_factor = 0.5
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        # Obtendo imagens do batch
        imgs = batch[0].reshape((-1, 28, 28, 1))
        
        # inserindo ruído aleatório
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Limitando os valores entre 0 e 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        
        # Definindo imagens com ruído como entradas e  originais como saídas
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,
                                                         targets_: imgs})

    print("Epoch: {}/{}...".format(e+1, epochs),
          "Training loss: {:.4f}".format(batch_cost))
    
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[:10]
noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = sess.run(decoded, feed_dict={inputs_: noisy_imgs.reshape((10, 28, 28, 1))})

for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)