import skrf as rf
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from fft import transform
from scipy import signal
from statistics import mean

def ifft(path, Zo=50):
    raw_points = 1001
    NFFT = 2**14
    beta = 0

    cable = rf.Network(path)

    # cable = cable.windowed()

    s11 = cable.s[:, 0, 0]
    print(s11)
    # s11 = cable.s11
    f = cable.frequency.f
    t_axis = np.linspace(0, 1 / cable.frequency.step, 101)
    # plt.plot(f, s11)
    # plt.show()

    # s11 = np.append(s11, np.zeros(NFFT-len(s11)))

    window_scale = 1.0 / (len(s11) * bessel0_ext(beta * beta / 4.0))
    window_scale = 1.0 / (2.0 * window_scale)
    # window = np.kaiser(NFFT, 0) # * window_scale
    # window = window_scale

    # s11 = [complex(d.re, d.im) for d in S11]
    window = np.blackman(len(s11))

    # s11 = window * s11

    # td = cable.s11.time_gate(center=0, span=.2, t_unit='ns')

    td = np.fft.ifft(s11, NFFT)
    # td = transform(s11, True)

    # s11 = cable.s[:, 0, 0]
    # window = np.kaiser(raw_points,6)
    # s11 = window * s11
    # td = np.abs(np.fft.ifft(s11, NFFT))

    # Create step waveform and compute step response
    step = np.ones(NFFT)
    step_response = np.convolve(td, step)
    step_response_Z = np.abs(Zo * (1 + step_response) / (1 - step_response))

    print(step_response_Z)
    # step_response_Z = step_response_Z[:1001]

    fs = 1000  # Sampling frequency
    # Generate the time vector properly
    fc = 2.3  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    tdk = signal.filtfilt(b, a, td)
    m = get_moving_average(tdk, round(len(tdk)/2))# mean(np.abs(tdk))

    m = np.mean(m)

    td_a = [x-m for x in td]
    td_m = [m for x in td_a]
    # Calculate maximum time axis
    t_axis = np.linspace(0, 1 / cable.frequency.step, NFFT)

    # td_r = np.zeros(len(td_a))
    td_r = [x for x in td_a]
    for i in range(1, len(td_r)):
        td_r[i] += td_r[i-1]

    # td_2 = [abs(x)*1000 for x in td]
    # td = [np.abs(x) for x in tdk]
    td_r = [2*x for x in td_r]



    return t_axis, td_r, td_m, step_response_Z

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


def get_moving_average(y=None, wsz=70):
    window_size = wsz

    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(y) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = y[i: i + window_size]

        # Calculate the average of current window
        window_average = sum(window) / window_size

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    u = []
    for i in range(len(moving_averages), len(y)):
        u.append(0)

    moving_averages = np.append(u, moving_averages)

    return moving_averages
