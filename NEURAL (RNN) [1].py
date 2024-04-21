import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd


veri= pd.read_csv("all_data.csv", delimiter=',', skiprows=0, low_memory=False)
print(veri.head())

print(veri['Label'].value_counts())
veri['Label'].replace(['BENIGN','DDoS','DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed'],[0,1,2,3,4,5,6],inplace=True)
print(veri['Label'].value_counts())
moddf = veri.dropna()


moddf.isna().any()[lambda x:x]

del moddf['Flow Packets/s']


features=["Destination Port" , "Bwd Packet Length Max", "Bwd Packet Length Mean" , "Bwd Packet Length Std", "Packet Length Mean",  "Average Packet Size", "Avg Bwd Segment Size","Flow Duration","Total Fwd Packets","Subflow Bwd Packets","Flow IAT Mean","Label"]

X = moddf[features].copy()
X.head()

giris = X.iloc[:,0:11]
cikis = X.iloc[:,-1]

print(cikis)

from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(giris, cikis, test_size=0.2)
xtrain

import tensorflow as tf
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
#Early stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)

x_train , y_train = np.array(xtrain), np.array(ytrain)
x_train = np.reshape(x_train, (xtrain.shape[0] , xtrain.shape[1], 1) )



y_train_cat = keras.utils.to_categorical(y_train)

keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=True)

regressor = Sequential()
regressor.add(Bidirectional(LSTM(units=11, return_sequences=True, input_shape = (x_train.shape[1],1) ) ))
regressor.add(LSTM(units= 10 , return_sequences=True))
regressor.add(LSTM(units= 10 , return_sequences=True))
regressor.add(LSTM(units= 10))
regressor.add(Dense(units = 7,activation='relu'))
regressor.compile(optimizer='SGD', loss='categorical_crossentropy',metrics=['acc'])

history = regressor.fit(x_train, y_train_cat, epochs=1,batch_size=128 )

trainloss=regressor.evaluate(x_train,y_train_cat,verbose=1)

x_test , y_test = np.array(xtest), np.array(ytest)
y_test_cat = keras.utils.to_categorical(y_test)
x_test = np.reshape(x_test, (xtest.shape[0] , xtest.shape[1], 1) )

testloss=regressor.evaluate(x_test,y_test_cat,verbose=1)

predictions = regressor.predict(x_test)

predictions

print(history.history.keys())

from keras.utils import plot_model
plot_model(regressor, show_shapes=True, to_file='model.png')

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()