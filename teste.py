
import skrf as rf
import matplotlib.pyplot as plt
rf.stylely()
from pylab import *

ntwk = rf.Network("C:\\Users\\Hardware\\Desktop\\git\\GuidedWavesDataAnalyzer\\s1p\\cinza\\CABO_CINZA.S1P")
ntwk_w = ntwk.windowed()
k = ntwk_w.plot_s_time_db()


plt.figure()
plt.title('Impulse Response Lowpass')
ntwk_w.s11.plot_s_time_impulse(pad=2000, window='hamming')


plt.figure()
plt.title('Impulse Response Bandpass')
ntwk_w.s11.plot_s_time(pad=2000, window='hamming')


plt.show(block=True)

