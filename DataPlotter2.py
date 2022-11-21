import csv
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from matplotlib.ticker import FormatStrFormatter
from ifft import ifft
from scipy.signal import butter, filtfilt
from pandas import DataFrame, date_range


class Data2:
    def __init__(self, guide_len, zo):
        self.dt_freq = DataFrame()
        self.dt_time = DataFrame()
        self.dt_sr = DataFrame()
        self.guide_len = guide_len
        self.Zo = zo

    def from_csv(self, path):
        pass

    def append_s1p(self, path, name):
        cable = rf.Network(path)
        freq = cable.frequency.f
        s11 = cable.s[:, 0, 0]

        if 'freq' in self.dt_freq.columns:
            pass
        else:
            self.dt_freq['freq'] = freq

        self.dt_freq[name] = s11

        t_axis, td_r, srz = ifft(cable, self.Zo)

        if 'time' in self.dt_time.columns:
            pass
        else:
            self.dt_time['time'] = t_axis
            self.dt_sr['time'] = t_axis

        self.dt_time[name] = td_r
        self.dt_sr[name] = srz
