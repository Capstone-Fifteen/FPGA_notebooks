import pynq.lib.dma
import numpy as np
from pynq import allocate
from pynq import Overlay
from pynq import Xlnk

def predictor(x):
    #quantise input
    x = (x * 255).astype(np.int32)
    #load overlay
    ol = Overlay("./design_2.bit")
    dma = ol.axi_dma_0
    with allocate(shape=(x.shape), dtype=np.int32) as input_buffer:
        input_buffer[:] = x
        with allocate(shape=(16,), dtype=np.int32) as output_buffer:
            dma.sendchannel.transfer(input_buffer)
            dma.recvchannel.transfer(output_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.wait()
            return np.argmax(output_buffer[0:9], axis=0)

import time

def measure_time(x):
    #quantise input
    x = (x * 255).astype(np.int32)
    #load overlay
    ol = Overlay("./design_2.bit")
    dma = ol.axi_dma_0
    with allocate(shape=(x.shape), dtype=np.int32) as input_buffer:
        input_buffer[:] = x
        with allocate(shape=(16,), dtype=np.int32) as output_buffer:
            start = time.time()
            dma.sendchannel.transfer(input_buffer)
            dma.recvchannel.transfer(output_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.wait()
            return time.time() - start