import skrf as rf
import matplotlib.pyplot as plt
from scipy import constants
import scipy as sp
import numpy as np
from fft import transform

from ctypes import *


def ifft(path, Zo=50):
    raw_points = 101
    NFFT = 256
    beta = 0

    cable = rf.Network(path)

    s11 = cable.s[:, 0, 0]

    s11 = np.append(s11, np.zeros(NFFT-len(s11)))

    window_scale = 1.0 / (NFFT * bessel0_ext(beta * beta / 4.0))

    window = np.kaiser(NFFT, 13) * window_scale
    window = window_scale
    s11 = window * s11
    td1 = sp.fft.ifft(s11, NFFT)
    td = transform(s11, True)

    # Create step waveform and compute step response
    step = np.ones(NFFT)
    step_response = np.convolve(td, step)
    step_response_Z = Zo * (1 + step_response) / (1 - step_response)
    step_response_Z = step_response_Z[:16384]

    td_r = [np.real(x) for x in td]
    td_i = [np.imag(x) for x in td]

    for i in range(1, len(td_r)):
        td_r[i] += td_r[i-1]

    # Calculate maximum time axis
    t_axis = np.linspace(0, 1 / cable.frequency.step, NFFT)

    # find the peak and distance
    # pk = np.max(td)
    # idx_pk = np.where(td == pk)[0]
    # print(d_axis[idx_pk[0]])

    return t_axis, td_r, step_response_Z


def bessel0_ext(x_pow_2):
    div = [1/4.0,   1/9.0,   1/16.0,
           1/25.0,  1/36.0,  1/49.0,
           1/64.0,  1/81.0,  1/100.0,
           1/121.0, 1/144.0, 1/169.0,
           1/196.0, 1/225.0, 1/256.0]

    SIZE = (13 - 2)
    i = SIZE
    term = x_pow_2
    ret = 1.0 + term

    '''
    do
    {
        term *= x_pow_2 * div[SIZE - i];
        ret += term;
    } while (--i);
    '''
    for i in range(0, SIZE):
        term *= x_pow_2 * div[i]
        ret += term

    return ret