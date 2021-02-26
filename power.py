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

rails = pynq.get_rails()
rail_names = ['PSINT_FP', 'PSINT_LP', 'PSPLL', 'PSDDR']

for rail in rail_names:
    print(f'Recording power for {rail}...')
    recorder = pynq.DataRecorder(rails[rail].power)
    with recorder.record(0.5):
        for i in range(0, 50):
            prediction = predictor(X[i])
    recorder.frame.to_csv(path_or_buf=rail+'.csv')

    


