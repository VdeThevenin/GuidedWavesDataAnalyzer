import skrf as rf
import scipy as sp
import numpy as np
from scipy import signal


def ifft(path, nfft=2**14):

    cable = rf.Network(path)
    s11 = cable.s[:, 0, 0]

    td = sp.fft.ifft(s11, nfft)

    # Create step waveform and compute step response
    step = np.ones(nfft)
    step_response = np.convolve(td, step)
    step_response_z = (1 + step_response) / (1 - step_response)

    fs = 1000
    fc = 2.3
    w = fc / (fs / 2)
    b, a = signal.butter(5, w, 'low')
    tdk = signal.filtfilt(b, a, td)
    m = get_moving_average(tdk, round(len(tdk)/2))
    m = np.mean(m)

    td_a = [x-m for x in td]
    t_axis = np.linspace(0, 1 / cable.frequency.step, nfft)

    td_r = [x for x in td_a]
    for i in range(1, len(td_r)):
        td_r[i] += td_r[i-1]

    td_r = [np.abs(2*np.real(x)) for x in td_r]

    return t_axis, td_r, step_response_z


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
