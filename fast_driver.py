import pynq.lib.dma
import time
import numpy as np
from pynq import allocate
from pynq import Overlay
from pynq import Xlnk

class Model:
    def __init__(self):
        self.input_buffer = allocate(shape=(90,), dtype=np.int32)
        self.output_buffer = allocate(shape=(16,), dtype=np.int32)
        self.ol = Overlay('./design_2.bit')

    def predict(self, x):
        #quantise input
        x = x.astype(np.int32)
        dma = self.ol.axi_dma_0
        self.input_buffer[:] = x
        dma.sendchannel.transfer(self.input_buffer)
        dma.recvchannel.transfer(self.output_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        return np.argmax(self.output_buffer[0:9], axis=0)

    def measure_time(self, x):
        #quantise input
        start = time.time()
        x = x.astype(np.int32)
        dma = self.ol.axi_dma_0
        self.input_buffer[:] = x
        dma.sendchannel.transfer(self.input_buffer)
        dma.recvchannel.transfer(self.output_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        return time.time() - start
