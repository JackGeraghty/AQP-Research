import numpy as np
import sys
import time
import math
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction

def apply_perturbation(result: dict, perturbation_function_name: str, perturbation_function_args: dict = None, **kwargs):
    perturbation_function = PERTURBATION_FUNCTIONS[perturbation_function_name]
    signal = result['signal']
    perturbed_signal = perturbation_function(signal, **perturbation_function_args, **kwargs)
    result[perturbation_function_args['out_key']] = perturbed_signal

def _add_noise(signal: np.ndarray, snr: float, **kwargs):
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
    
    return noisy

def to_polar(complex_ar):
    return np.abs(complex_ar),np.angle(complex_ar)

def _join_signals(signal_one, signal_two):
    return 

def _add_babble(signal: np.ndarray):
    return 


def find_all_available_functions():
    current_module = sys.modules[__name__]
    members = getmembers(current_module, isfunction)
    print(members)

PERTURBATION_FUNCTIONS =  {"_add_noise": _add_noise}# find_all_available_functions()

