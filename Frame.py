from Snapshot import Snapshot
from pandas import DataFrame
from Cable import Cable
import matplotlib.pyplot as plt


class Frame:
    def __init__(self, zo, length):
        self.timeline = []
        self.x = []
        self.tdr = DataFrame()
        self.z_response = DataFrame()
        self.average = [[], []]
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
        self.z_response[name] = ss.get_z_response()
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
        for i in range(0, len(tdr)-1):
            deriv.append((tdr[i + 1] - tdr[i]) / (t[i + 1] - t[i]))
            max_deriv = max(deriv)
            if deriv[-1] == max_deriv:
                imax = i

        self.dtdr_dt = deriv
        self.cable.set_tbreak(t[imax])

        # self.cable.calculate_params()

    def set_x(self):
        t = self.timeline
        v = self.cable.speed

        self.x = [ti*v/2 for ti in t]

    def run(self):
        self.mean()
        self.derivate()
        self.set_x()

    def plot(self, figure):
        figure.suptitle(self.name, x=0.2, y=0.98)
        plt.tight_layout()
        ax = figure.add_subplot(1, 1, 1)

        ax.plot(self.x, self.average[0], label=self.name)

        ax.grid()
        ax.grid(visible=True, which='minor', color='lightgray', linestyle='-')

        handles, labels = ax.get_legend_handles_labels()

        ax.legend(handles[::-1], labels[::-1])

        ax.set_xlabel('time [s]')
        ax.set_ylabel(r'$S_{11}$' + " Parameter")

    def update(self, c, l, s, rp):
        self.cable.speed = s
        self.cable.capacitance = c
        self.cable.inductance = l
        self.cable.relative_permissivity = rp
        self.run()

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

