from bluepy.btle import DefaultDelegate, UUID, Peripheral, BTLEException, BTLEDisconnectError
from helper import DataPacket, MyArduino, corrupted_packet, LeftRightManager
import json
import sys
from tensorflow import keras
import numpy as np
import time

filename = './example.txt'
if (len(sys.argv) > 1):
    filename = sys.argv[1]

dataCounter = 0
dancePosition = 1
raw_data = [[], [], [], [], [] ,[]]
leftright = LeftRightManager()

#update your model path here
model = keras.models.load_model('./model/left_right_model_1')
w = 10
t = 48

def rolling_minmax(arr, window_size):
    result = []
    for i in range(0, len(arr) - window_size):
        local_min = min(arr[i: i + window_size])
        local_max = max(arr[i: i + window_size])
        result.append(local_max - local_min)
    return result

# filter/rolling mean, set n=1, w is window size
def convolve(arr, w, n):
    if n == 0:
        return arr
    return convolve(np.convolve(arr, np.ones(w), 'valid') / w, w, n - 1)

# rolling mean feature (measure average xyzypr values)
def feature_1(arr):
    arr = convolve(arr, t + 1, 1)
    return arr

def feature_2(arr, spike_size=150):
    arr = rolling_minmax(arr, t)
    arr = [spike_size if x > spike_size else x for x in arr]
    return arr

def feature_3(arr, step=2, stability_factor=t):
    if (len(arr) < stability_factor):
        return arr
    arr = np.diff(arr, n=1)
    arr = np.absolute(arr)
    arr = convolve(arr, stability_factor, 1)
    arr = [0 if x < step else x for x in arr]
    return arr

# jerk/delta function
def delta(arr):
    return np.diff(np.array(arr), n=1)

def prediction():
    final_features = []
    final_features.append(feature_1(raw_data[0]))
    final_features.append(feature_1(raw_data[1]))
    final_features.append(feature_1(raw_data[2]))
    final_features.append(feature_1(raw_data[4]))
    final_features.append(feature_1(raw_data[5]))
    final_features.append(feature_2(raw_data[3], 100))
    final_features.append(feature_2(raw_data[4]))
    final_features.append(feature_3(raw_data[0], 0))
    final_features.append(feature_3(raw_data[1], 0))
    final_features.append(feature_3(raw_data[2], 0))
    final_features.append(feature_3(raw_data[4], 0))
    final_features.append(feature_3(raw_data[5], 0))

    u = [-2115.534613970284,
    1745.0986129472676,
    -731.1806362298021,
    -74.24628997431007,
    31.276243747701713,
    39.52484568863395,
    39.68763638630838,
    114.07980417523952,
    132.33187464929236,
    90.21077529771183,
    1.7870775921192095,
    1.8900713365338655]

    s = [1528.6806359796387,
    1485.3316546967924,
    1297.8928765369553,
    57.92389451706283,
    51.73119557546073,
    29.734565552817795,
    38.51798709728382,
    67.24541300002035,
    92.56493343101354,
    63.16309455693329,
    1.6874139787493947,
    1.5788794320350645]
    for i in range(0, len(final_features)):
        final_features[i] = (np.array(final_features[i]) - u[i]) / s[i]

    in_data = [[]]
    for i in range(0, len(final_features[0])):
        for j in range(0, len(final_features)):
            in_data[0].append(final_features[j][i])
    p = model.predict(np.array(in_data))
    return np.argmax(p[0])

def handleDataPacket(packet, mac_address_index):
    global dataCounter
    global raw_data
    global leftright

    dancePositionMask = dancePosition << 118
    packet = packet | dancePositionMask  # set dance position bits

    decodedPacket = DataPacket(packet)
    
    if corrupted_packet(decodedPacket):
        return
    if filename:
        with open(filename, "a") as myfile:
            myfile.write(str(decodedPacket.xAccel) +  ',' + str(decodedPacket.yAccel) + ',' + str(decodedPacket.zAccel) + ',' + str(decodedPacket.yaw) + ',' + str(decodedPacket.pitch) + ',' + str(decodedPacket.row) + '\n')
    
    raw_data[0].append(decodedPacket.xAccel)
    raw_data[1].append(decodedPacket.yAccel)
    raw_data[2].append(decodedPacket.zAccel)
    raw_data[3].append(decodedPacket.yaw)
    raw_data[4].append(decodedPacket.pitch)
    raw_data[5].append(decodedPacket.row)

    if len(raw_data[0]) == 20:
        p = leftright.getDirection(raw_data)
        with open('./prediction.csv', 'a') as f:
            f.write(str(p) + '\n')
        raw_data = [[], [], [], [], [] ,[]]

    

    '''
    if len(raw_data[0]) == t + w:
        p = prediction()
        with open('./prediction.csv', 'a') as f:
            f.write(str(p) + '\n')
        raw_data = [[], [], [], [], [] ,[]]
    '''


myArduino = MyArduino(mac_address_index=1, handleDataPacket=handleDataPacket)
myArduino.run()
