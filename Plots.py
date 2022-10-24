# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:39:33 2022

@author: Renato
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import random

si = []
ti = []
tick = [0, 0]
v = 0
def main():
    names_n_paths = []
    
    #subpath = "C:\\Users\\Renato\\Desktop\\TCC\\"
    
    subpath = "C:\\Users\\Vidya-HW2\\Desktop\\TCC\\cvs\\"
    
    names_n_paths.append(["RG-6 - INITIAL CONDITION",       subpath + "RG6_2M_ALUM_ORIG.CSV"])
    names_n_paths.append(["RG-6 - CUTS",                    subpath + "RG6_2M_ALUM_T0.5M.CSV"])
    names_n_paths.append(["RG-6 - CUTS AND CORROSIONS",     subpath + "RG6_2M_ALUM_CORROSIONS.CSV"])
    names_n_paths.append(["RG-59 - INITIAL CONDITION",      subpath + "RG59_2.92M_ORIG.CSV"])
    names_n_paths.append(["RG-59 - CUTS",                   subpath + "RG59_2.92M_T0.5M.CSV"])
    names_n_paths.append(["RG-59 - CUTS AND CORROSIONS",    subpath + "RG59_2.92M_CORROSIONS.CSV"])
    names_n_paths.append(["RG-59 - LONG CUT 1",             subpath + "RG59_2.92M_CORTE7.CSV"])
    names_n_paths.append(["RG-59 - LONG CUT 2",             subpath + "RG59_2.92M_CORTE8.CSV"])
    
    for nap in names_n_paths:
        import_data_and_print(nap[0], nap[1])
    
    names_n_paths = []
    
    names_n_paths.append(["RG-59 -- CORTE 1", subpath + "RG59_2.92M_CORTE1.CSV", "0% loss"])
    names_n_paths.append(["RG-59 -- CORTE 2", subpath + "RG59_2.92M_CORTE2.CSV", "20% loss"])
    names_n_paths.append(["RG-59 -- CORTE 3", subpath + "RG59_2.92M_CORTE3.CSV", "40% loss"])
    names_n_paths.append(["RG-59 -- CORTE 4", subpath + "RG59_2.92M_CORTE4.CSV", "60% loss"])
    names_n_paths.append(["RG-59 -- CORTE 5", subpath + "RG59_2.92M_CORTE5.CSV", "80% loss"])
    names_n_paths.append(["RG-59 -- CORTE 6", subpath + "RG59_2.92M_CORTE6.CSV", "100% loss"])
    
    print_live("CUT FROM 0 TO 100%", names_n_paths)
    
    names_n_paths = []
    
    names_n_paths.append(["RG-59 -- CORROSION 1", subpath + "RG59_2.92M_CORROSION_FINAL.CSV", "initial"])             #20:32:00
    names_n_paths.append(["RG-59 -- CORROSION 2", subpath + "RG59_2.92M_CORROSION_FINAL2.CSV", "90 s"])         #20:33:34
    names_n_paths.append(["RG-59 -- CORROSION 3", subpath + "RG59_2.92M_CORROSION_FINAL3.CSV", "180 s"])        #20:34:56
    names_n_paths.append(["RG-59 -- CORROSION 4", subpath + "RG59_2.92M_CORROSION_FINAL4.CSV", "240 s"])        #20:36:08
    names_n_paths.append(["RG-59 -- CORROSION 5", subpath + "RG59_2.92M_CORROSION_FINAL5.CSV", "300 s"])        #20:37:10
    names_n_paths.append(["RG-59 -- CORROSION 6", subpath + "RG59_2.92M_CORROSION_FINAL6.CSV", "340 s"])        #20:37:48
    names_n_paths.append(["RG-59 -- CORROSION 7", subpath + "RG59_2.92M_CORROSION_FINAL7.CSV", "400 s"])        #20:38:50
    names_n_paths.append(["RG-59 -- CORROSION 8", subpath + "RG59_2.92M_CORROSION_FINAL8.CSV", "washed"])        #20:50:08
    
    print_live("CORROSION FROM 0 TO 100%", names_n_paths)
    
def import_data_and_print(name, csvpath, figure):
    global si
    global ti
    global v

    ret = []

    with open(csvpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        t = []
        s = []
    
        
        for row in csv_reader:
            if len(row) < 3 or row[0]=="Time":
                pass
            else:                
                t.append(float(row[0]))
                s.append(float(row[1]))
    
    if "INITIAL" in name: 
        si = s
        ti = t

    
    mt = []
    ms = [] 
    msi = []
    window_size = 70
  
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
      
    # Loop through the array to consider
    # every window of size 3
    while i < len(s) - window_size + 1:
        
        # Store elements from i to i+window_size
        # in list to get the current window
        window = s[i : i + window_size]
      
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
          
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
          
        # Shift window to right by one position
        i += 1
     
    
    if "--" not in name:
        lim = 0.0050
    else:
        lim = 0.005
    
    deriv = []
    
    imax = 0
    for i in range(int(len(s)/10), len(s)-1):
        deriv.append((si[i+1]-si[i])/(ti[i+1]-ti[i]))
        maxderiv = max(deriv)
        if deriv[-1] == maxderiv:
            imax = i

    t_break = ti[imax]
    s_break = si[imax]
    
    if "RG-6" in name: 
        length = 2.0
    else:
        length = 2.92

    #wave_speed
    v = (2*length)/t_break
      

    if "INITIAL" in name: 
        
        
        z0 = 75
        
        C = 1/(z0*v)
        
        L = 1/(C*v**2)
        
        e0 = 8.85e-12
        u0 = 4*np.pi*1e-7
        
        er = 1/(v**2*e0*u0)
        
        print(name)
        print(f'L={L} H')
        print(f'C={C} F')
        print(f'er={er:0.2f}')
        print(f'v={v} m/s')
        print()
    
 
    t_lim = 0.02e-8
    
    for i in range(int(len(s)/10), imax):
        if s[i] > s[i-1]:
            if s[i] > s[i+1]:
                diff = s[i] - si[i]
                if diff > lim:
                    if len(mt) > 0:
                        dt = t[i] - mt[-1]
                    else:
                        dt = 1
                    if dt > t_lim:
                        mt.append(t[i])    
                        ms.append(s[i])
                        msi.append(s[i] - si[i])
                    else:
                        mt[-1] = t[i]    
                        ms[-1] = s[i]
                        msi[-1] = s[i] - si[i]
       
    positions = [v*tp/2 for tp in mt] 
    
    tick_size = 100
    
    major_ticks = np.arange(0, tick_size+1, 20)
    minor_ticks = np.arange(0, tick_size+1, 5)
    
    global tick
    if "INITIAL" in name:
        tick[0] = t[-1]/tick_size
        tick[1] = s[-1]/tick_size
    
    major_ticks_t = [x*tick[0] for x in major_ticks] 
    minor_ticks_t = [x*tick[0] for x in minor_ticks] 
    
    major_ticks_s = [x*tick[1] for x in major_ticks] 
    minor_ticks_s = [x*tick[1] for x in minor_ticks] 
    fig = figure
    # fig = plt.figure()
    fig.suptitle(name, x=0.2, y=0.98)
    plt.tight_layout()
    ax = fig.add_subplot(1, 1, 1)
    
    
    ax.plot(t,s, label="current")
    ax.plot(ti, si, label="initial")
    # ax.plot(t,mean)
    
    plt.xlim([0, tick[0]])
    
    arrowprops = dict(
    arrowstyle = "->")
    
    for i in range(0, len(mt)):
        r = random.randint(3,7)/10
        ax.annotate(f"{ms[i]:.2f}\n@{positions[i]:.2f} m", xy = (mt[i],ms[i]), xytext =(mt[i],r), ha='center', fontsize=5,arrowprops = arrowprops)
        if "INITIAL" not in name: 
            r = random.randint(0,1)/10
            ax.annotate(f"diff\n{msi[i]:.2E}", xy = (mt[i],0.2), xytext =(mt[i],r), ha='center', fontsize=4,arrowprops = arrowprops)
     

    ax.annotate('break', xy = (t_break,s_break), xytext =(t_break,s_break+0.2),arrowprops = arrowprops)
    
    def tick_fct(v, tick):
        return ["%.2f" % (v*tk/2) for tk in tick]
    
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(major_ticks_t)
    ax2.set_xticklabels(tick_fct(v, major_ticks_t))
    ax2.set_xticks(minor_ticks_t, minor=True)
    
    ax2.set_xlabel("length [m]")
    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'$S_{11}$' + " Parameter")
   
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    subpath = "C:\\Users\\Vidya-HW2\\Desktop\\TCC\\graph\\"
    #subpath = "C:\\Users\\Renato\\Desktop\\TCC\\graphs\\"

    ax.set_xticks(major_ticks_t)
    ax.set_xticks(minor_ticks_t, minor=True)
    ax.set_yticks(major_ticks_s)
    ax.set_yticks(minor_ticks_s, minor=True)
    ax.grid()
    ax.grid(visible=True, which='minor', color='lightgray', linestyle='-')
    
    handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    ax.legend(handles[::-1], labels[::-1])

    
    plt.savefig(subpath + name +'.eps', format='eps')
    plt.show()
   

def print_live(name, names_n_paths):
    global tick       
    
    fig = plt.figure()
    fig.suptitle(name,x=0.2, y=0.98)
    ax = fig.add_subplot(1, 1, 1)
    
    lbl = ""
    lblint = 0
    
    for nnp in names_n_paths:
        csvpath = nnp[1]
        with open(csvpath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            t = []
            s = []
            
            
            for row in csv_reader:
                if len(row) < 3 or row[0]=="Time":
                    pass
                else:             
                    if float(row[0]) > 2.5e-8:
                       
                        t.append(float(row[0]))
                        s.append(float(row[1]))
                    
        lbl = nnp[2]
                
        ax.plot(t,s, label=lbl)
        # ax.plot(t,mean)
        
        
                
        tick_size = 100
        
        major_ticks = np.arange(80, tick_size+1, 20)
        minor_ticks = np.arange(80, tick_size+1, 5)
        
       
        tick[0] = t[-1]/tick_size
        tick[1] = s[-1]/tick_size
        
        plt.xlim([t[0], t[-1]])
        
        major_ticks_t = [x*tick[0] for x in major_ticks] 
        minor_ticks_t = [x*tick[0] for x in minor_ticks] 
        
        major_ticks_s = [x*tick[1] for x in major_ticks] 
        minor_ticks_s = [x*tick[1] for x in minor_ticks] 
        

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
    
    subpath1 = "C:\\Users\\Vidya-HW2\\Desktop\\TCC\\graph\\"
    plt.savefig(subpath1 + name +'.eps',format='eps', bbox_extra_artists=[name])
    plt.show()
    
    j = 1