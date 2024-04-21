import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import time
seconds = time.time()

dtypes = {'Src IP': 'category', 'Src Port': 'uint16', 'Dst IP': 'category', 'Dst Port': 'uint16', 'Protocol': 'float32', 'Flow Duration': 'uint32', 'Tot Fwd Pkts': 'uint32', 'Tot Bwd Pkts': 'uint32', 'TotLen Fwd Pkts': 'float32', 'TotLen Bwd Pkts': 'float32', 'Fwd Pkt Len Max': 'float32', 'Fwd Pkt Len Min': 'float32', 'Fwd Pkt Len Mean': 'float32', 'Fwd Pkt Len Std': 'float32', 'Bwd Pkt Len Max': 'float32', 'Bwd Pkt Len Min': 'float32', 'Bwd Pkt Len Mean': 'float32', 'Bwd Pkt Len Std': 'float32', 'Flow Byts/s': 'float32', 'Flow Pkts/s': 'float32', 'Flow IAT Mean': 'float32', 'Flow IAT Std': 'float32', 'Flow IAT Max': 'float32', 'Flow IAT Min': 'float32', 'Fwd IAT Tot': 'float32', 'Fwd IAT Mean': 'float32', 'Fwd IAT Std': 'float32', 'Fwd IAT Max': 'float32', 'Fwd IAT Min': 'float32', 'Bwd IAT Tot': 'float32', 'Bwd IAT Mean': 'float32', 'Bwd IAT Std': 'float32', 'Bwd IAT Max': 'float32', 'Bwd IAT Min': 'float32', 'Fwd PSH Flags': 'float32', 'Bwd PSH Flags': 'float32', 'Fwd URG Flags': 'float32', 'Bwd URG Flags': 'float32', 'Fwd Header Len': 'uint32', 'Bwd Header Len': 'uint32', 'Fwd Pkts/s': 'float32', 'Bwd Pkts/s': 'float32', 'Pkt Len Min': 'float32', 'Pkt Len Max': 'float32', 'Pkt Len Mean': 'float32', 'Pkt Len Std': 'float32', 'Pkt Len Var': 'float32', 'FIN Flag Cnt': 'float32', 'SYN Flag Cnt': 'float32', 'RST Flag Cnt': 'float32', 'PSH Flag Cnt': 'float32', 'ACK Flag Cnt': 'float32', 'URG Flag Cnt': 'float32', 'CWE Flag Count': 'float32', 'ECE Flag Cnt': 'float32', 'Down/Up Ratio': 'float32', 'Pkt Size Avg': 'float32', 'Fwd Seg Size Avg': 'float32', 'Bwd Seg Size Avg': 'float32', 'Fwd Byts/b Avg': 'uint32', 'Fwd Pkts/b Avg': 'uint32', 'Fwd Blk Rate Avg': 'uint32', 'Bwd Byts/b Avg': 'uint32', 'Bwd Pkts/b Avg': 'uint32', 'Bwd Blk Rate Avg': 'uint32', 'Subflow Fwd Pkts': 'uint32', 'Subflow Fwd Byts': 'uint32', 'Subflow Bwd Pkts': 'uint32', 'Subflow Bwd Byts': 'uint32', 'Init Fwd Win Byts': 'uint32', 'Init Bwd Win Byts': 'uint32', 'Fwd Act Data Pkts': 'uint32', 'Fwd Seg Size Min': 'uint32', 'Active Mean': 'float32', 'Active Std': 'float32', 'Active Max': 'float32', 'Active Min': 'float32', 'Idle Mean': 'float32', 'Idle Std': 'float32', 'Idle Max': 'float32', 'Idle Min': 'float32', 'Label': 'category'}
df = pd.read_csv("final_dataset.csv", dtype = dtypes, usecols= [*dtypes.keys()], engine = 'c', low_memory= True)
print(df.describe(include='all'))
print("Потраченное количество времени: = ", time.time()-seconds, "секунд")
#columnsToDrop = np.array(['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])

#col = np.union1d(columnsToDrop, ['Src IP', 'Dst IP'])
col = np.array(['Src IP', 'Dst IP'])
df.drop(columns = col, inplace = True)
df.replace([np.inf, -np.inf], np.nan, inplace=True) # Убираем бесконечно большие и малые числа , заменив их на значения NaN
df = df.fillna(0)      # Убираем строки со значением NaN
df['Label'].replace(['Benign','ddos'],[1,0],inplace=True)         # Превращение выходных значений в бинарный вид

print(df.shape)
features = ["Bwd Pkt Len Std", "Dst Port", "Flow Pkts/s", "Fwd Pkt Len Max", "Src Port", "Tot Bwd Pkts", "Label"]
#logfeatures = ["Dst Port", "Fwd Pkt Len Max", "Bwd Pkt Len Min", 'Flow IAT Mean', "Fwd Pkt Len Std", "Fwd Pkt Len Mean", "TotLen Fwd Pkts", "Src Port", "Fwd Pkt Len Min", "Tot Fwd Pkts", "Flow Pkts/s", "Flow Duration", "Flow Byts/s", "Label"]
#X = df.copy()
#features = ["Bwd Pkt Len Std", "Label"]
X = df.copy()
print(X.head())

del df
x_train = X.iloc[:,:-1]
y_train = X.iloc[:,-1]

from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(x_train, y_train, test_size=0.2)

import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras import callbacks
from tensorflow import keras

x_train , y_train = np.array(xtrain), np.array(ytrain)
print(x_train.shape)
#Реализовать Early Stopping
MLP = Sequential()
# MLP.add(Dense(units = 50, activation='relu', input_shape=(x_train.shape[1], )))
# MLP.add(Dense(units = 50, activation= 'relu'))
# MLP.add(Dense(units = 50, activation= 'relu'))
# MLP.add(Dense(units = 1, activation= 'softmax'))
# MLP.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
MLP.add(LSTM(units = 10, input_shape = (x_train.shape[1], 1,), return_sequences = True))
MLP.add(LSTM(units = 10, return_sequences = True))
MLP.add(LSTM(units = 10))
MLP.add(Dense(units = 1, activation= 'sigmoid'))
MLP.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = MLP.fit(x_train, y_train, epochs=10 ,batch_size=128, validation_split= 0.3, verbose=1)

train_loss, train_acc = MLP.evaluate(x_train, y_train,verbose=1)

x_test , y_test = np.array(xtest), np.array(ytest)
test_loss, test_acc = MLP.evaluate(x_test, y_test, verbose=1)
print()
print(f'Тестирование точности предсказания: {test_acc}', end = "\n\n")

print("Потраченное количество времени: = ", time.time()-seconds, "секунд", end = "\n\n")
#Реализовать выборочную тестовую выборку

print("Введите 50 случайных строк для ручной выборки: ")

for i in range(50):
    z = int(input())
    predictions = MLP.predict(X.iloc[z:z+1, :-1])
    print(f'Строка: {z}  Ответ: {predictions}   Правильный ответ: {X.iloc[z, -1]}')

#Shift + Alt + E

import neptune

run = neptune.init_run(
    project="falearn/falearn",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vbmV3LXVpLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNTkzZTg5OC00ZWY0LTRiZjItYjhkZS1jMzY2ZjU2NjJiZWQifQ==",
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)

run["eval/f1_score"] = 0.66
run.stop()