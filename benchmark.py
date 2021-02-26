import pynq
import numpy as np
import time
from driver import predictor
from numpy import genfromtxt

print('Importing csv data...')
data = genfromtxt('./input300.csv', delimiter=' ')
X = data.transpose()[:-1].transpose().astype('float32')
values = data.transpose()[-1].astype(int)
n_values = np.max(values) + 1
y = np.eye(n_values)[values]

total_time = 0
incorrect = 0
start_time = time.time()

for i in range(0, 500):
    actual = predictor(X[i])
    expected = np.argmax(y[i], axis=0)
    if expected != actual:
        incorrect += 1
        print(f'Incorrect output for test case {i}\nExpected: {expected}\nActual: {actual}')

total_time = time.time() - start_time
accuracy = 100.0 - incorrect / 5.0
print(f'Accuracy: {accuracy}\nTotal time: {total_time}')

from driver import measure_time

fpga_time = 0
for i in range(0, 500):
    fpga_time += measure_time(X[i])
print(f'FPGA time: {fpga_time}')
'''
#unavailable yet
from tensorflow import keras
import tensorflow as tf
prediction = model.predict(X)

software_time = time.time()
model = keras.models.load_model('./my_model')
software_time = time.time() - software_time
m = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)
m.update_state(prediction, y)

factor = software_time / prediction_time
print(f'Speed up factor: {factor}\nSoftware time: {software_time}\nFPGA time: {prediction_time}\n');
'''