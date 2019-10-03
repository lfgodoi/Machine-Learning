# Keras: 2-D Convolutional Neural Network (2-D CNN) para Prognóstico de Motor Turbofan

#------------------------------------------------------------------------------

##### Importando pacotes e subpacotes necessários #####

# Pandas: pacote para manipulação e análise de dados
# NumPy: pacote para arrays e matrizes multidimensionais
# Scikit-Learn: pacote de machine learning
# Keras: pacote de redes neurais
# Matplotlib: pacote de plotagem
# Itertools: módulo contendo ferramentas para manipulação de iterações

import pandas as pd 
import numpy as np
import itertools
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#------------------------------------------------------------------------------

##### Seleção de turbofan #####
choose = str(input('Escolha um dataset:\n\t (1) - FD001 (tamanho: 20631 / motores: 100) \n\t (2) - FD002 (tamanho: 73759 / motores: 259) \n\t (3) - FD003 (tamanho: 24720 / motores: 100) \n\t (4) - FD004 (tamanho: 61249 / motores: 248)\n'))
tf_train = './train_FD00' + choose + '.txt'
tf_test = './test_FD00' + choose + '.txt'
tf_RUL = './RUL_FD00' + choose + '.txt'
print('Dataset escolhido -> FD00' + choose)

##### Carregando os dados de treinamento #####

# Lendo o arquivo TXT no padrão CSV (Comma-Separated Values) com identificador automático de separador
train_df = pd.read_csv(tf_train, sep=" ", header=None)
# Eliminando as colunas 26 e 27 do DataFrame retornado
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
# Atribuindo rótulos às colunas
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
# Mostrando a ID do DataFrame
print('#id:',len(train_df.id.unique()))
# Ordenando o conjunto por ID e ciclo            
train_df = train_df.sort_values(['id','cycle'])
# Mostrando as dimensões do DataFrame
print(train_df.shape)
# Três primeiras linhas do DataFrame
train_df.head(3)

#------------------------------------------------------------------------------

##### Plotando as frequências de treinamento #####

# criando uma figura de dimensões 20x6 pol
plt.figure(figsize=(20,6))
# Plotando um gráfico com a frequência de cada valor
train_df.id.value_counts().plot.bar()
# Frequência média do conjunto
print("medium working time:", train_df.id.value_counts().mean())
# Frequência máxima do conjunto
print("max working time:", train_df.id.value_counts().max())
# Frequência mínima do conjunto
print("min working time:", train_df.id.value_counts().min())

#------------------------------------------------------------------------------

##### Plotando os dados dos sensores do motor #####

# Coletando os valores pertencentes à ID especificada
engine_id = train_df[train_df['id'] == 1]
# Plotando oo gráfico correspondente à cada sensor
ax1 = engine_id[train_df.columns[2:]].plot(subplots=True, sharex=True, figsize=(20,30))

#------------------------------------------------------------------------------

##### Carregando os dados de teste #####

# Lendo o arquivo TXT no padrão CSV (Comma-Separated Values) com identificador automático de separador
test_df = pd.read_csv(tf_test, sep=" ", header=None)
# Eliminando as colunas 26 e 27 do DataFrame retornado
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
# Atribuindo rótulos às colunas
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
# Mostrando a ID do DataFrame
print('#id:',len(test_df.id.unique()))
# Mostrando as dimensões do DataFrame
print(train_df.shape)
# Três primeiras linhas do DataFrame
train_df.head(3)

#------------------------------------------------------------------------------

##### Carregando os cados de validação (confiança) #####

# Lendo o arquivo TXT no padrão CSV (Comma-Separated Values) com identificador automático de separador
truth_df = pd.read_csv(tf_RUL, sep=" ", header=None)
# Eliminando a segunda coluna do DataFrame retornado
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
truth_df.columns = ['more']
# Ajusta a contagem dos índices das linhas (0,1,... -> 1,2,...)
truth_df = truth_df.set_index(truth_df.index + 1)
# Mostrando as dimensões do DataFrame
print(truth_df.shape)
# Três primeiras linhas do DataFrame
truth_df.head(3)

#------------------------------------------------------------------------------

##### Calculando o treinamento da RUL #####

# 
train_df['RUL']=train_df.groupby(['id'])['cycle'].transform(max)-train_df['cycle']
train_df.RUL[0:10]

#------------------------------------------------------------------------------

##### ADD NEW LABEL TRAIN #####
w1 = 45
w0 = 15
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

#------------------------------------------------------------------------------

##### Redimensionando os dados de treinamento #####

def scale(df):
    return (df - df.min())/(df.max()-df.min())

for col in train_df.columns:
    if col[0] == 's':
        train_df[col] = scale(train_df[col])
        
train_df = train_df.dropna(axis=1)
train_df.head()

#------------------------------------------------------------------------------

##### Calcula teste de RUL #####
truth_df['max'] = test_df.groupby('id')['cycle'].max() + truth_df['more']
test_df['RUL'] = [truth_df['max'][i] for i in test_df.id] - test_df['cycle']

#------------------------------------------------------------------------------

##### Adiciona novo rótulo de treinamento #####
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

#------------------------------------------------------------------------------

##### Redimensionando os dados de teste #####

for col in test_df.columns:
    if col[0] == 's':
        test_df[col] = scale(test_df[col])
        
test_df = test_df.dropna(axis=1)
test_df.head()

#------------------------------------------------------------------------------

##### Gerador de sequência #####

sequence_length = 50

def gen_sequence(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 (start stop) -> from row 0 to row 50
    # 1 51 (start stop) -> from row 1 to row 51
    # 2 52 (start stop) -> from row 2 to row 52
    # ...
    # 141 191 (start stop) -> from row 141 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
        
def gen_labels(id_df, seq_length, label):

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]

#------------------------------------------------------------------------------

##### Sequência de colunas: Colunas a considerar #####
sequence_cols = []
for col in train_df.columns:
    if col[0] == 's':
        sequence_cols.append(col)
#sequence_cols.append('cycle_norm')
print(sequence_cols)

#------------------------------------------------------------------------------

##### Gerando teste do treinamento X #####
x_train, x_test = [], []
for engine_id in train_df.id.unique():
    for sequence in gen_sequence(train_df[train_df.id==engine_id], sequence_length, sequence_cols):
        x_train.append(sequence)
    for sequence in gen_sequence(test_df[test_df.id==engine_id], sequence_length, sequence_cols):
        x_test.append(sequence)
    
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

print("X_Train shape:", x_train.shape)
print("X_Test shape:", x_test.shape)

#------------------------------------------------------------------------------

##### Gerando teste do treinamento Y #####
y_train, y_test = [], []
for engine_id in train_df.id.unique():
    for label in gen_labels(train_df[train_df.id==engine_id], sequence_length, ['label2'] ):
        y_train.append(label)
    for label in gen_labels(test_df[test_df.id==engine_id], sequence_length, ['label2']):
        y_test.append(label)
    
y_train = np.asarray(y_train).reshape(-1,1)
y_test = np.asarray(y_test).reshape(-1,1)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#------------------------------------------------------------------------------

##### Codificando o rótulo #####
y_train = np_utils.to_categorical(y_train)
print(y_train.shape)

y_test = np_utils.to_categorical(y_test)
print(y_test.shape)

#------------------------------------------------------------------------------

##### Gerando imagens #####

from scipy.spatial.distance import pdist, squareform

def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

plt.figure(figsize=(20,20))
for i in range(0,17):
    
    plt.subplot(6, 3, i+1)    
    rec = rec_plot(x_train[0,:,i])
    plt.imshow(rec)
    plt.title(sequence_cols[i])
plt.show()

#------------------------------------------------------------------------------

##### Transformando os testes do treinamento X em imagens
x_train_img = np.apply_along_axis(rec_plot, 1, x_train).astype('float16')
print(x_train_img.shape)

x_test_img = np.apply_along_axis(rec_plot, 1, x_test).astype('float16')
print(x_test_img.shape)

#------------------------------------------------------------------------------

##### Modelo #####

filepath = "./RUL_CNN.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max', save_weights_only=False)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 17)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

BATCH_SIZE = 200
EPOCHS = 10

history = model.fit(x_train_img,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=[checkpoint],
                      validation_split=0.2,
                      verbose=1)

#------------------------------------------------------------------------------

##### Carregando o melhor modelo #####
best_model = load_model(filepath)

model.evaluate(x_test_img, y_test, verbose=1)[1]

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

print(classification_report(np.where(y_test != 0)[1], model.predict_classes(x_test_img)))

cnf_matrix = confusion_matrix(np.where(y_test != 0)[1], model.predict_classes(x_test_img))
plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=np.unique(np.where(y_test != 0)[1]), title="Confusion matrix")
plt.show()

#------------------------------------------------------------------------------
