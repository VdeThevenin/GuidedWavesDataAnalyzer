from Snapshot import Snapshot
from pandas import DataFrame
from Cable import Cable
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class Frame:
    def __init__(self, zo, length):
        self.td_wo_os = []
        self.td_w_os = []
        self.raw_freq = []
        self.d2rdt2 = []
        self.debug = []
        self.timeline = []
        self.x = []
        self.tdr = DataFrame()
        self.raw_s11 = DataFrame()
        self.z_response = DataFrame()
        self.imax = 0
        self.ibreak = 0
        self.average = [[], [], []]
        self.dtdr_dt = []
        self.cable = Cable(zo, length)
        self.name = ""

    def add_snapshot(self, ss: Snapshot):
        if len(self.timeline) == 0:
            self.timeline = ss.get_time()

        name = ss.get_name()
        if self.name == "":
            self.name = name

        self.tdr[name] = ss.get_s11()
        self.raw_s11[name] = ss.s1p[0]
        self.raw_freq = ss.s1p[2]
        self.td_w_os = ss.td_w_offset
        self.td_wo_os = ss.td_wo_offset
        self.z_response[name] = np.abs(50 * ss.get_z_response())
        self.mean()

    def add_snapshots_list(self, sslist):
        for ss in sslist:
            self.add_snapshot(ss)

    def mean(self):
        self.average[0] = self.tdr.mean(axis=1)
        self.average[1] = self.z_response.mean(axis=1)
        self.average[2] = self.raw_s11.mean(axis=1)

    def derivate(self):
        deriv = []
        t = self.timeline
        tdr = self.average[0]

        for i in range(0, min([len(tdr), len(t)]) - 1):
            deriv.append((tdr[i + 1] - tdr[i]) / (t[i + 1] - t[i]))

        deriv.append(0)

        self.dtdr_dt = [d / max(deriv) for d in deriv]

    def set_x(self):
        t = self.timeline
        v = self.cable.speed

        self.x = np.array([ti * v / 2 for ti in t])

    def run(self):
        self.mean()

        self.average[0] = convolve_with_rect(self.timeline, self.average[0], 10)
        self.debug.append([self.timeline, self.average[0], 'tdr'])
        self.derivate()
        self.set_ibreak()

    def set_ibreak(self, ib=None):
        if ib is None:
            self.get_ibreak()
        else:
            self.ibreak = ib + 10
            self.imax = self.ibreak + 10
        self.cable.set_tbreak(self.timeline[self.ibreak])
        self.cable.set_vbreak(self.average[0][self.ibreak])
        self.set_x()

    def get_ibreak(self):
        rect_len = int(len(self.average[0]) / 6)
        rmp = convolve_with_rect(self.timeline, self.average[0], rect_len)
        drdt = derivate(self.timeline, self.average[0])
        self.debug.append([self.timeline, drdt, 'drdt'])
        self.d2rdt2 = derivate(self.timeline, drdt)
        self.debug.append([self.timeline, self.d2rdt2, 'd2rdt2'])

        pair = ramp_detector(self.average[0])
        tnl, nl_curve = nameless_function(self.timeline, pair, self.d2rdt2)
        # self.debug.append([tnl, nl_curve, 'nl'])
        for p in pair:
            h = 0.5  # self.average[0][p[0]]
            rect = do_rect(self.timeline, p[0], p[1], height=h)
            self.debug.append([self.timeline, rect, ''])

        idx = np.where(nl_curve == max(nl_curve))

        ib = np.where(self.timeline == tnl[idx[0][0]])
        self.ibreak = ib[0][0] + 10
        self.imax = self.ibreak + 10

    def update(self, c, l, s, rp, ib=None):

        if ib is not None:
            self.set_ibreak(ib)
        self.cable.speed = s
        self.cable.capacitance = c
        self.cable.inductance = l
        self.cable.relative_permissivity = rp
        self.run()


def plot_frames(frames, figure, q=None, debug=True):
    assert (len(frames) > 0)
    name = "Reflection by Guide Length"
    figure.suptitle(name, x=0.2, y=0.98)
    plt.tight_layout()
    ax = figure.add_subplot(1, 1, 1)

    ilim = frames[0].imax

    if debug:
        for pair in frames[-1].debug:
            if pair[2] == '':
                plt.fill_between(pair[0], pair[1], step="pre", alpha=0.3)
            ax.plot(pair[0], pair[1], label=pair[2])

        ib = frames[-1].ibreak
        ax.scatter(frames[-1].timeline[ib], frames[-1].average[0][ib], color='black')

        plt.figure(2)
        plt.stem(frames[0].raw_freq, frames[0].average[2])
        plt.xscale('log')
        plt.grid()

        plt.figure(3)
        plt.plot(frames[0].timeline, frames[0].td_w_os)
        # plt.plot(frames[0].timeline, frames[0].td_wo_os)
        plt.grid()

        plt.figure(4)
        plt.plot(frames[0].timeline, frames[0].average[0])
        # plt.plot(frames[0].timeline, frames[0].td_wo_os)
        plt.grid()

        plt.show()


    else:
        x = frames[0].x
        if q is None:
            ax.plot(x[:ilim], frames[0].average[0][:ilim], label="ref: " + str(frames[0].name))

            if len(frames) > 2:
                ax.plot(x[:ilim], frames[-2].average[0][:ilim], label="last: " + str(frames[-2].name))
            if len(frames) > 1:
                ax.plot(x[:ilim], frames[-1].average[0][:ilim], label="new: " + str(frames[-1].name))

            t0, x0, y0 = frame_compare(frames[0], frames[-1], frames[0].ibreak)
            if len(x0) > 0:
                ax.scatter(x0[:ilim], y0[:ilim], label="detected", c='red')
                for i in range(0, len(x0)):
                    ax.annotate(f'{y0[i]:.2f}\n@{x0[i]:.2f}',
                                xy=(x0[i], y0[i]),  # theta, radius
                                xytext=(x0[i], y0[i] + 0.2),  # fraction, fraction
                                arrowprops=dict(facecolor='black', shrink=0.005),
                                horizontalalignment='center',
                                verticalalignment='top')

            plt.axvline(x[frames[0].ibreak], color='red', label='break')
        else:
            if q == 1:
                ax.plot(x[:ilim], frames[0].average[0][:ilim], label="sample")
                plt.axvline(frames[0].x[frames[0].ibreak], color='red', label='break')
            else:
                ax.plot(x, frames[0].average[0], label="ref: " + str(frames[0].name))
                for i in range(1, min([q, len(frames)]) - 1):
                    ax.plot(frames[i].x, frames[i].average[0],
                            label="frame #" + str(i + 1) + ": " + str(frames[i].name))
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
    s = (value / abs(value))
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

    a = s * a

    if m >= 0:
        return a, (m_arr[m] + main_unit)
    else:
        return a, (d_arr[d] + main_unit)


def frame_compare(f0: Frame, f1: Frame, ilim):
    dmin = 0.25
    diffmin = 0.005
    x = f0.x[:ilim]
    t = f0.timeline[:ilim]
    y0 = f0.average[0][:ilim]
    y1 = f1.average[0][:ilim]
    dy1 = f1.dtdr_dt[:ilim]
    d2y1 = derivate(x, dy1)
    d2y1 = [-d for d in d2y1]
    d2y1 = d2y1[:ilim]
    indexes = []
    for i in range(1, len(d2y1) - 1):
        if d2y1[i - 1] < d2y1[i] > d2y1[i + 1] and d2y1[i] > dmin and (y1[i] - y0[i]) > diffmin:
            indexes.append(i)

    return np.array(t[indexes]), np.array(x[indexes]), np.array(y1[indexes])


def derivate(x, y):
    deriv = []
    t = x
    tdr = y

    imax = 0
    for i in range(0, len(tdr) - 1):
        deriv.append((tdr[i + 1] - tdr[i]) / (t[i + 1] - t[i]))
        max_deriv = max(deriv)
        if deriv[-1] == max_deriv:
            imax = i

    for i in range(len(deriv), len(tdr)):
        deriv.append(0)

    deriv = [d / max(deriv) for d in deriv]

    return deriv


def convolve_with_rect(t, y, rect_len):
    rect = do_rect(t, rect_len, 2 * rect_len)

    return convolve(rect, y, rect_len)


def do_rect(t, start, stop, height=1):
    T = t[1] - t[0]
    rect = np.where(np.logical_and(t >= start * T, t <= stop * T), height, 0)
    return rect


def convolve(y1, y2, x_offset):
    n0 = x_offset
    n = len(y1)
    y1 = np.convolve(y1, y2, mode='full')  # scaled convolution

    return np.array([k / max(y1[n0:n]) for k in y1[n0:n + n0]])


def correlate(y1, y2):
    n = len(y1)
    y1 = signal.correlate(y1, y2, mode='same') / (
            np.linalg.norm(y1) * np.linalg.norm(y2))  # np.correlate(y1, y2, mode='full')  # scaled convolution

    return np.array([k for k in y1[:n]])


def nameless_function(t, p, dy):
    nl = []
    t0 = []
    for i in range(0, len(p)):
        d = p[i][1] - p[i][0]
        h = dy[p[i][0]]
        v = d * h
        nl.append(v)
        t0.append(t[p[i][0]])
    return t0, nl


def greatest_ramp_detector(curve, t):
    ramps = ramp_detector(curve)

    index_greatest_width = 0
    greatest_width = 0
    for i in range(0, len(ramps)):
        diff = ramps[i][1] - ramps[i][0]
        if diff >= greatest_width:
            greatest_width = diff
            index_greatest_width = i

    curve2 = []
    for p in ramps:
        curve2.append((curve[p[1]] - curve[p[0]]) / (t[p[1]] - t[p[0]]))

        curve2 = [c2 / max(curve2) for c2 in curve2]

    return ramps[index_greatest_width][0], ramps, curve2


def ramp_detector(curve):
    points = []
    start = -1
    stop = -1
    percent = [0.015, 0.015]
    ramp_min_width = 5
    init = 0  # int(len(curve) / 10)
    for i in range(init, len(curve)):
        if curve[i] >= curve[i - 1] * (1 + percent[0]):
            if start < 0:
                start = i
            else:
                stop = i
        else:
            if curve[i] > curve[i - 1] * (1 - percent[1]):
                if stop - start >= ramp_min_width:
                    points.append([start, stop])
                    start = -1
                    stop = -1

    return points


def get_peaks_idxs(d):
    d = np.abs(d)
    dmin = 0.5
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
    for i in range(0, len(idxs) - 1):
        areas.append(peak_area([idxs[i], idxs[i + 1]], tdr))

    areas = np.array([a / max(areas) for a in areas])

    idxs = np.array(idxs[:-1])

    return idxs, areas
