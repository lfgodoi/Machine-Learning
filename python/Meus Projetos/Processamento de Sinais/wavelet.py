# Wavelet

import pandas as pd
import pywt
dir_health = 'health_30_2.csv'
dataset_health = pd.read_csv(dir_health, sep='\t', header=-1)
dataset_train = dataset_health[:600000]

dataset_train = dataset_train.iloc[:, :-1]
dataset_train.columns = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
                         'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']

cA, cD = pywt.dwt(dataset_train,'haar')
import matplotlib.pyplot as plt

