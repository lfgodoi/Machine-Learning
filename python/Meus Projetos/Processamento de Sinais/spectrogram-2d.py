# Geração de espectrograma 2-D

# Importando os pacotes e módulos
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Carregando e manipulando o dataset
dir_health = 'health_30_2.csv'
dataset_health = pd.read_csv(dir_health, sep='\t', header=-1)
dataset_train = dataset_health[:600000]
dataset_train = dataset_train.iloc[:, :-1]
dataset_train.columns = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
                         'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']

# Gerando o espectrograma
fs = 10e3
f, t, Sxx = signal.spectrogram(dataset_train.iloc[:,1], fs)
spec = plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
cbar = plt.colorbar(spec)
plt.show()
