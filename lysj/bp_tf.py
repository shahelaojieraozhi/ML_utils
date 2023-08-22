# -*- coding: utf-8 -*-
"""
@Project ：ML_utils 
@Time    : 2023/8/22 18:19
@Author  : Rao Zhi
@File    : bp_tf.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('E:\\File_cache\\feiq\\Recv Files\\data_9_120.csv')

data = dataset.iloc[:, :9].values
label = dataset.iloc[:, -2].values

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.7)

x_train = scaler.fit_transform(X_train)
x_test = scaler.fit_transform(X_test)
num_features = 9


y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)

model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=0.02), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2)

print(np.argmax(model.predict(x_test), axis=1))
a = y_test
y_predict = np.argmax(model.predict(x_test), axis=1)
print("The accuracy of test data set is：", accuracy_score(y_test, y_predict))
model.save('model.h5')
