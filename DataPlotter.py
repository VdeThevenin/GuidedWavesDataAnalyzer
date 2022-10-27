# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:59:03 2022

@author: Vidya-HW2
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class Data:
    def __init__(self, csvpath, name, guide_len=1.0, Zo=75.0):
        with open(csvpath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            t = []
            y = []
        
            
            for row in csv_reader:
                if len(row) < 3 or row[0]=="Time":
                    pass
                else:                
                    t.append(float(row[0]))
                    y.append(float(row[1]))
        
        self.name = name
        self.t = t
        self.y = y
        self.x = 0
        self.guide_len = guide_len
        self.Zo = Zo
        
        self.mAvg = 0 
        self.deriv = 0
        self.t_break = 0
        self.y_break = 0
        self.imax = 0
        self.diff_positions = []
        
        self.inductance = 0
        self.capacitance = 0
        self.speed = 0
        self.relative_permissivity = 0
        
        self.get_moving_average()
        self.get_deriv()
        self.calculate_params()
        self.set_x()
        
    def get_moving_average(self, wsz = 70):
        window_size = wsz
        y = self.y
        
        i = 0
        # Initialize an empty list to store moving averages
        moving_averages = []
          
        # Loop through the array to consider
        # every window of size 3
        while i < len(y) - window_size + 1:
            
            # Store elements from i to i+window_size
            # in list to get the current window
            window = y[i : i + window_size]
          
            # Calculate the average of current window
            window_average = round(sum(window) / window_size, 2)
              
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
              
            # Shift window to right by one position
            i += 1
        return moving_averages
    
    def get_deriv(self, lim_i=None, lim_s=None):
        t = self.t
        y = self.y
        
        if lim_i is None or lim_s is None:
            lim_i = int(len(y)/10)
            lim_s = len(y)-1
        
        deriv = []
        
        imax = 0
        for i in range(lim_i, lim_s):
            deriv.append((y[i+1]-y[i])/(t[i+1]-t[i]))
            maxderiv = max(deriv)
            if deriv[-1] == maxderiv:
                imax = i

        self.imax = imax
        self.t_break = t[imax]
        self.y_break = y[imax]
        self.deriv = deriv
        
        
    def calculate_params(self):

        if self.guide_len == 0:
            self.guide_len = 1e-25
        if self.Zo == 0:
            self.Zo = 1e-25
        self.speed = (2*self.guide_len)/self.t_break
                    
        self.capacitance = 1/(self.Zo*self.speed)
        
        self.inductance = 1/(self.capacitance*self.speed**2)
        
        e0 = 8.85e-12
        u0 = 4*np.pi*1e-7
        
        self.relative_permissivity = 1/(self.speed**2*e0*u0)

    def set_x(self):
        t = self.t
        v = self.speed

        self.x = [ti*v/2 for ti in t]

    def set_Zo(self, Zo):
        self.Zo = Zo
        self.calculate_params()
        
    def set_GuideLen(self, guide_len):
        self.guide_len = guide_len
        self.calculate_params()
        
    def diff(self, data : 'Data', lim_i=None, lim_s=None, threshold=0.005, window=5):

        y = self.y
        y0 = data.y
        
        if lim_i is None or lim_s is None:
            lim_i = int(len(y)/10)+window
            lim_s = len(y) - window
            
        diff_positions = []
            
        for i in range(lim_i, lim_s):
            diff = y[i] - y0[i]
            if diff > threshold:
                if isTheBiggestInWindow(i, y, window):
                    diff_positions.append(i)
                    
        self.diff_positions = diff_positions
        
        
def isTheBiggestInWindow(i, y, window):
    
    window_arr = y[i-int(window/2):i-1] + y[i+1: i+int(window/2)]
    
    r = True
    
    for e in window_arr:
        if y[i] < e:
            r = False
    return r


def plot_array(data_arr, name, figure):
    figure.suptitle(name, x=0.2, y=0.98)
    plt.tight_layout()
    ax = figure.add_subplot(1, 1, 1)

    for data in data_arr:
        ax.plot(data.t, data.y, label=data.name)

    ax.grid()
    ax.grid(visible=True, which='minor', color='lightgray', linestyle='-')

    handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    ax.legend(handles[::-1], labels[::-1])

    #ax2 = ax.twiny()
    #ax2.set_xlim(ax.get_xlim())
    # ax2.set_xticks(data_arr[0].t)
    # ax2.set_xticklabels(data_arr[0].t)
    # ax2.set_xticks(minor_ticks_t, minor=True)

    #ax2.set_xlabel("length [m]")
    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'$S_{11}$' + " Parameter")

def plot_data(ref: Data, data : Data, figure):
    tick = [0, 0]
    tick_size = 100
    
    major_ticks = np.arange(0, tick_size+1, 20)
    minor_ticks = np.arange(0, tick_size+1, 5)

    tick[0] = ref.x[-1]/tick_size
    tick[1] = ref.y[-1]/tick_size
    
    major_ticks_t = [x*tick[0] for x in major_ticks] 
    minor_ticks_t = [x*tick[0] for x in minor_ticks] 
    
    major_ticks_s = [x*tick[1] for x in major_ticks] 
    minor_ticks_s = [x*tick[1] for x in minor_ticks]

    figure.suptitle(data.name, x=0.2, y=0.98)
    plt.tight_layout()
    ax = figure.add_subplot(1, 1, 1)

    ax.plot(data.x, data.y, label="current")
    ax.plot(ref.x, ref.y, label="initial")
    # ax.plot(t,mean)
    
    plt.xlim([0, tick[0]])
    
    
    def tick_fct(v, tick):
        return ["%.2f" % (v*tk/2) for tk in tick]
    
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(major_ticks_t)
    ax2.set_xticklabels(tick_fct(data.speed, major_ticks_t))
    ax2.set_xticks(minor_ticks_t, minor=True)

    ax2.set_xlabel("length [m]")
    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'$S_{11}$' + " Parameter")
   
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_xticks(major_ticks_t)
    ax.set_xticks(minor_ticks_t, minor=True)
    ax.set_yticks(major_ticks_s)
    ax.set_yticks(minor_ticks_s, minor=True)
    ax.grid()
    ax.grid(visible=True, which='minor', color='lightgray', linestyle='-')
    
    handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    ax.legend(handles[::-1], labels[::-1])


def set_unit_prefix(value, main_unit):

    m_arr = ['k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    d_arr = ['m', '\u03BC', 'n', 'p', 'f', 'a', 'z', 'y']

    a = abs(value)
    s = (value/abs(value))
    m = -1
    d = -1

    if a >= 1000:
        while a >= 1000:
            a /= 1000
            m += 1
    elif a <= 0.001:
        while a <= 1:
            a *= 1000
            d += 1
    else:
        return value, main_unit

    a = s*a

    if m >= 0:
        return a, (m_arr[m]+main_unit)
    else:
        return a, (d_arr[d]+main_unit)
