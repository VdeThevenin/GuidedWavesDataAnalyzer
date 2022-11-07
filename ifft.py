import skrf as rf
import matplotlib.pyplot as plt
from scipy import constants
import numpy as np

def ifft_new():
    raw_points = 101
    NFFT = 16384
    PROPAGATION_SPEED = 66.8
    Zo = 75

    _prop_speed = PROPAGATION_SPEED / 100
    cable = rf.Network('C:\\Users\\Vidya-HW2\\Desktop\\TCC\\cable_r59.s1p')

    s11 = cable.s[:, 0, 0]
    window = np.hanning(raw_points)
    s11 = window * s11
    td1 = np.fft.ifft(s11, NFFT)

    td = np.abs(np.fft.ifft(s11, NFFT))

    td = td * 1000

    # Create step waveform and compute step response
    step = np.ones(NFFT)
    step_response = np.convolve(td1, step)
    step_response_Z = Zo * (1 + step_response) / (1 - step_response)
    step_response_Z = step_response_Z[:16384]

    # Calculate maximum time axis
    t_axis = np.linspace(0, 1 / cable.frequency.step, NFFT)
    d_axis = constants.speed_of_light * _prop_speed * t_axis / 2

    # find the peak and distance
    pk = np.max(td)
    idx_pk = np.where(td == pk)[0]
    # print(d_axis[idx_pk[0]])

    return t_axis[:idx_pk[0]], td[:idx_pk[0]], step_response_Z[:idx_pk[0]]

    # Plot time response
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax2.set_ylim([0, 500])
    # ax2.yaxis.set_ticks(np.arange(0, 500, 50))
    # ax1.plot(t_axis[:idx_pk[0]], td[:idx_pk[0]], 'g-')
    # ax2.plot(t_axis[:idx_pk[0]], step_response_Z[:idx_pk[0]], 'r-')
    # ax1.set_xlabel("Distance (m)")
    # ax1.set_ylabel("Reflection Magnitude")
    # ax2.set_ylabel("Impedance (Ohms)")
    # ax1.set_title("Return loss Time domain")
    # plt.show()