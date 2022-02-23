import numpy as np
import time
import math
from .node import AQPNode


class PerturbationNode(AQPNode):
    
    def __init__(self, id_, output_key, target_key, snr, draw_options=None, **kwargs):
        super().__init__(id_, output_key=output_key, draw_options=draw_options, **kwargs)
        self.target_key = target_key
        self.snr = snr
        self.type_ = "PerturbationNode"
    
    def execute(self, result, **kwargs):
        signal = result[self.target_key]
        snr = self.snr
        
        np.random.seed((int) (time.time()))
        rms_signal = math.sqrt(np.mean(signal**2))
        rms_noise = math.sqrt(rms_signal**2/(pow(10,snr/10)))
        STD_n=rms_noise
        noise=np.random.normal(0, STD_n, signal.shape[0])
            
        noisy = signal + noise
            
            # X=np.fft.rfft(noise)
            # radius,angle=to_polar(X)
            # plt.plot(radius)
            # plt.xlabel("FFT coefficient")
            # plt.ylabel("Magnitude")
            # plt.show()
            # signal_noise=signal+noise
            # plt.plot(signal_noise, label="noisy", color='r')
            # plt.plot(signal, label="pure", alpha=0.66)
            # plt.legend()
            # plt.title(f'SNR {snr}')
            # plt.ylabel("Amplitude")
            # plt.savefig(f'snr_{snr}.jpg')
            # plt.show()
        result[self.output_key] = noisy
        return result