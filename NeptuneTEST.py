import tensorflow as tf
from keras.layers import LSTM, Dense
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git

run = neptune.init_run(
    project="falearn/falearn",
    api_token="",
)  # your credentials

params = {"lr": 0.001, "epochs": 8, "batch_size": 128}
run["parameters"] = params

dtypes = {'Src IP': 'category', 'Src Port': 'uint16', 'Dst IP': 'category', 'Dst Port': 'uint16', 'Protocol': 'float32', 'Flow Duration': 'uint32', 'Tot Fwd Pkts': 'uint32', 'Tot Bwd Pkts': 'uint32', 'TotLen Fwd Pkts': 'float32', 'TotLen Bwd Pkts': 'float32', 'Fwd Pkt Len Max': 'float32', 'Fwd Pkt Len Min': 'float32', 'Fwd Pkt Len Mean': 'float32', 'Fwd Pkt Len Std': 'float32', 'Bwd Pkt Len Max': 'float32', 'Bwd Pkt Len Min': 'float32', 'Bwd Pkt Len Mean': 'float32', 'Bwd Pkt Len Std': 'float32', 'Flow Byts/s': 'float32', 'Flow Pkts/s': 'float32', 'Flow IAT Mean': 'float32', 'Flow IAT Std': 'float32', 'Flow IAT Max': 'float32', 'Flow IAT Min': 'float32', 'Fwd IAT Tot': 'float32', 'Fwd IAT Mean': 'float32', 'Fwd IAT Std': 'float32', 'Fwd IAT Max': 'float32', 'Fwd IAT Min': 'float32', 'Bwd IAT Tot': 'float32', 'Bwd IAT Mean': 'float32', 'Bwd IAT Std': 'float32', 'Bwd IAT Max': 'float32', 'Bwd IAT Min': 'float32', 'Fwd PSH Flags': 'float32', 'Bwd PSH Flags': 'float32', 'Fwd URG Flags': 'float32', 'Bwd URG Flags': 'float32', 'Fwd Header Len': 'uint32', 'Bwd Header Len': 'uint32', 'Fwd Pkts/s': 'float32', 'Bwd Pkts/s': 'float32', 'Pkt Len Min': 'float32', 'Pkt Len Max': 'float32', 'Pkt Len Mean': 'float32', 'Pkt Len Std': 'float32', 'Pkt Len Var': 'float32', 'FIN Flag Cnt': 'float32', 'SYN Flag Cnt': 'float32', 'RST Flag Cnt': 'float32', 'PSH Flag Cnt': 'float32', 'ACK Flag Cnt': 'float32', 'URG Flag Cnt': 'float32', 'CWE Flag Count': 'float32', 'ECE Flag Cnt': 'float32', 'Down/Up Ratio': 'float32', 'Pkt Size Avg': 'float32', 'Fwd Seg Size Avg': 'float32', 'Bwd Seg Size Avg': 'float32', 'Fwd Byts/b Avg': 'uint32', 'Fwd Pkts/b Avg': 'uint32', 'Fwd Blk Rate Avg': 'uint32', 'Bwd Byts/b Avg': 'uint32', 'Bwd Pkts/b Avg': 'uint32', 'Bwd Blk Rate Avg': 'uint32', 'Subflow Fwd Pkts': 'uint32', 'Subflow Fwd Byts': 'uint32', 'Subflow Bwd Pkts': 'uint32', 'Subflow Bwd Byts': 'uint32', 'Init Fwd Win Byts': 'uint32', 'Init Bwd Win Byts': 'uint32', 'Fwd Act Data Pkts': 'uint32', 'Fwd Seg Size Min': 'uint32', 'Active Mean': 'float32', 'Active Std': 'float32', 'Active Max': 'float32', 'Active Min': 'float32', 'Idle Mean': 'float32', 'Idle Std': 'float32', 'Idle Max': 'float32', 'Idle Min': 'float32', 'Label': 'category'}
df = pd.read_csv("final_dataset.csv", dtype = dtypes, usecols= [*dtypes.keys()], engine = 'c', low_memory= True)
col = np.array(['Src IP', 'Dst IP'])
df.drop(columns = col, inplace = True)
df.replace([np.inf, -np.inf], np.nan, inplace=True) # Убираем бесконечно большие и малые числа , заменив их на значения NaN
df = df.fillna(0)      # Убираем строки со значением NaN
df['Label'].replace(['Benign','ddos'],[1,0],inplace=True)   # Превращение выходных значений в бинарный вид

logfeatures = ["Dst Port", "Fwd Pkt Len Max", "Bwd Pkt Len Min", 'Flow IAT Mean', "Fwd Pkt Len Std", "Fwd Pkt Len Mean", "TotLen Fwd Pkts", "Src Port", "Tot Fwd Pkts", "Label"]
features = ["Bwd Pkt Len Std", "Dst Port", "Label"]
X = df[logfeatures].copy()

del df
x_train = X.iloc[:,:-1]
y_train = X.iloc[:,-1]

from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(x_train, y_train, test_size=0.1)
x_train , y_train = np.array(xtrain), np.array(ytrain)
x_test , y_test = np.array(xtest), np.array(ytest)
MLP = tf.keras.models.Sequential(
    [
        LSTM(units=10, input_shape=(x_train.shape[1], 1,), return_sequences=True),
        LSTM(units=10, return_sequences=True),
        LSTM(units=10),
        Dense(units=1, activation='sigmoid'),
        # Dense(units = 50, input_shape = (x_train.shape[1], ), activation='relu'),
        # Dense(units = 50, activation='relu'),
        # Dense(units = 50, activation='relu'),
        # Dense(units=1, activation='sigmoid'),
    ]
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=params["lr"],
)

MLP.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)

neptune_cbk = NeptuneCallback(run=run, base_namespace="training")

MLP.fit(
    x_train,
    y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[neptune_cbk],
    validation_split = 0.3
)
x_test , y_test = np.array(xtest), np.array(ytest)
test_loss, test_acc = MLP.evaluate(x_test, y_test, verbose=1)
print()
print(f'Тестирование точности предсказания: {test_acc}', end = "\n\n")

from sklearn.metrics import confusion_matrix
x_test, xteeest, y_test, yteeest = tts(x_test, y_test, test_size=0.1)
y_true = np.array(yteeest, dtype = np.int16)
y_pred = MLP.predict(xteeest)
y_pred = np.array(y_pred, dtype = np.int16)
print(y_pred, y_test)
dc_mf_tn, dc_mf_fp, dc_mf_fn, dc_mf_tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel()
print(f'Всего данных: {len(y_true)}')
print(f'Количество ошибок при предсказании DDoS атак: {dc_mf_fn}')

eval_metrics = MLP.evaluate(x_test, y_test, verbose=0)
for j, metric in enumerate(eval_metrics):
    run["eval/{}".format(MLP.metrics_names[j])] = metric

run.stop()
