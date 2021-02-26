import numpy as np
from numpy import genfromtxt

data = genfromtxt('./input300.csv', delimiter=' ')
X = data.transpose()[:-1].transpose().astype('float32')
values = data.transpose()[-1].astype(int)
n_values = np.max(values) + 1
y = np.eye(n_values)[values]

print(X.shape)
print(y.shape)

from pynq import allocate
from pynq import Overlay
import pynq.lib.dma
from pynq import Xlnk
import time

prediction = []
acc_time = 0
prediction_time = 0
start_time = time.time()
incorrect = 0

for l in range(0, 5):
    ol = Overlay("./design_2.bit")
    dma = ol.axi_dma_0
    with allocate(shape=(X.shape[1]), dtype=np.int32) as input_buffer:
        #Z = []
        for i in range(0, len(X[l])):
            input_buffer[i] = int(X[l][i] * 255)
            #Z.append(input_buffer[i])
        with allocate(shape=(16,), dtype=np.int32) as output_buffer:
            a_time = time.time()
            dma.sendchannel.transfer(input_buffer)
            dma.recvchannel.transfer(output_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.wait()
            print(output_buffer)
            fpga_out = np.argmax(output_buffer[0:9], axis=0)
            real_out = np.argmax(y[l], axis=0)
            if fpga_out != real_out:
                incorrect += 1
                print(f'Expected: {real_out}\nActual: {fpga_out}')
            output_buffer.freebuffer()
            prediction_time += time.time() - a_time
        input_buffer.freebuffer()

stop_time = time.time()
acc_time += stop_time - start_time
accuracy = 100.0 - incorrect / 5.0
print(f'Accuracy: {accuracy}\nTotal time taken: {acc_time}\nPrediction time taken: {prediction_time}')
