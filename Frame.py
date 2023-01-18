from Snapshot import Snapshot
from pandas import DataFrame
from Cable import Cable
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class Frame:
    def __init__(self, zo, length):
        self.timeline = []
        self.x = []
        self.tdr = DataFrame()
        self.z_response = DataFrame()
        self.imax = 0
        self.ibreak = 0
        self.average = [[], []]
        self.dtdr_dt = []
        self.idxs = []
        self.conv_avg = []
        self.cable = Cable(zo, length)
        self.name = ""

    def add_snapshot(self, ss: Snapshot):
        if len(self.timeline) == 0:
            self.timeline = ss.get_time()

        name = ss.get_name()
        if self.name == "":
            self.name = name

        self.tdr[name] = ss.get_s11()
        self.z_response[name] = np.abs(50*ss.get_z_response())
        self.mean()

    def add_snapshots_list(self, sslist):
        for ss in sslist:
            self.add_snapshot(ss)

    def mean(self):
        self.average[0] = self.tdr.mean(axis=1)
        self.average[1] = self.z_response.mean(axis=1)

    def derivate(self):
        deriv = []
        t = self.timeline
        tdr = self.average[0]

        imax = 0
        for i in range(0, min([len(tdr), len(t)])-1):
            deriv.append((tdr[i + 1] - tdr[i]) / (t[i + 1] - t[i]))

        deriv.append(0)

        self.dtdr_dt = [d/max(deriv) for d in deriv]

        self.idxs, areas = get_peak_areas(tdr, self.dtdr_dt)

        imax = self.idxs[np.where(areas == max(areas))]
        imax = imax[0]
        self.imax = imax
        self.ibreak = imax-20
        self.cable.set_tbreak(t[self.ibreak])
        self.cable.set_vbreak(tdr[self.ibreak])

    def set_x(self):
        t = self.timeline
        v = self.cable.speed

        self.x = np.array([ti*v/2 for ti in t])

    def run(self):
        self.mean()
        self.derivate()
        self.set_x()
        self.conv_avg = convolve_with_rect(self.timeline, self.average[0], 10)

    def update(self, c, l, s, rp):
        self.cable.speed = s
        self.cable.capacitance = c
        self.cable.inductance = l
        self.cable.relative_permissivity = rp
        self.run()


def plot_frames(frames, figure, q=None):
    assert(len(frames) > 0)
    name = "Reflection by Guide Length"
    figure.suptitle(name, x=0.2, y=0.98)
    plt.tight_layout()
    ax = figure.add_subplot(1, 1, 1)

    ilim = frames[0].imax
    conv = correlate(frames[0].average[0], frames[-1].average[0])
    if q is None:
        ax.plot(frames[0].x[:ilim], frames[0].average[0][:ilim], label="ref: " + str(frames[0].name))
        # ax.plot(frames[0].x[:ilim], frames[0].average[1][:ilim], label="z")
        if len(frames) > 2:
            ax.plot(frames[-2].x[:ilim], frames[-2].average[0][:ilim], label="last: " + str(frames[-2].name))
        if len(frames) > 1:
            ax.plot(frames[-1].x[:ilim], frames[-1].average[0][:ilim], label="new: " + str(frames[-1].name))

        x0, y0 = frame_compare(frames[0], frames[-1], ilim)
        ax.scatter(x0[:ilim], y0[:ilim], label="detected", c='red')

        plt.axvline(frames[0].x[frames[0].ibreak], color='red', label='break')
    else:
        if q == 1:
            ax.plot(frames[0].x[:ilim], frames[0].average[0][:ilim], label="sample")
            plt.axvline(frames[0].x[frames[0].ibreak], color='red', label='break')
        else:
            ax.plot(frames[0].x, frames[0].average[0], label="ref: " + str(frames[0].name))
            for i in range(1, min([q, len(frames)])-1):
                ax.plot(frames[i].x, frames[i].average[0], label="frame #" + str(i+1) + ": " + str(frames[i].name))
            ax.plot(frames[-1].x, frames[-1].average[0], label="new: " + str(frames[-1].name))
    ax.grid()
    ax.grid(visible=True, which='minor', color='lightgray', linestyle='-')

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles[::-1], labels[::-1])

    ax.set_xlabel('cable length [m]')
    ax.set_ylabel(r'$S_{11}$ [V]')

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


def frame_compare(f0:Frame, f1:Frame, ilim):
    dmin = 0.25
    diffmin = 0.02
    x = f1.x[:ilim]
    y0 = f0.average[0][:ilim]
    y1 = f1.average[0][:ilim]
    dy1 = f1.dtdr_dt[:ilim]
    d2y1 = derivate(x, dy1)
    d2y1 = [-d for d in d2y1]
    d2y1 = d2y1[:ilim]
    indexes = []
    for i in range(1, len(d2y1)-1):
        if d2y1[i-1] < d2y1[i] > d2y1[i+1] and d2y1[i] > dmin and (y1[i]-y0[i]) > diffmin:
            indexes.append(i)

    return np.array(x[indexes]), np.array(y1[indexes])


def derivate(x, y):
    deriv = []
    t = x
    tdr = y

    imax = 0
    for i in range(0, len(tdr)-1):
        deriv.append((tdr[i + 1] - tdr[i]) / (t[i + 1] - t[i]))
        max_deriv = max(deriv)
        if deriv[-1] == max_deriv:
            imax = i

    for i in range(len(deriv), len(tdr)):
        deriv.append(0)

    deriv = [d/max(deriv) for d in deriv]

    return deriv


def convolve_with_rect(t, y, rect_len):
    st = rect_len
    T = t[1] - t[0]  # sampling width
    rect = np.where(np.logical_and(t >= st*T, t <= (st+rect_len)*T), 1, 0)  # build input functions

    return convolve(rect, y)


def convolve(y1, y2):
    n = len(y1)
    y1 = np.convolve(y1, y2, mode='full')  # scaled convolution

    return np.array([k / max(y1[:n]) for k in y1[:n]])


def correlate(y1, y2):
    n = len(y1)
    y1 = signal.correlate(y1, y2, mode='same')/(np.linalg.norm(y1) * np.linalg.norm(y2)) # np.correlate(y1, y2, mode='full')  # scaled convolution

    return np.array([k for k in y1[:n]])


def get_peaks_idxs(d):
    d = np.abs(d)
    dmin = 0.3
    idxs = []
    for i in range(1, len(d) - 1):
        if d[i - 1] < d[i] > d[i + 1] and d[i] > dmin:
            idxs.append(i)
    return idxs


def peak_area(idxs: tuple, tdr):
    a = 0
    for i in range(idxs[0], idxs[1]):
        a += tdr[i]

    return a


def get_peak_areas(tdr, dydt):
    idxs = get_peaks_idxs(dydt)

    idxs.append(len(tdr))
    areas = []
    for i in range(0, len(idxs)-1):
        areas.append(peak_area([idxs[i], idxs[i+1]], tdr))

    areas = np.array([a/max(areas) for a in areas])

    idxs = np.array(idxs[:-1])

    return idxs, areas
