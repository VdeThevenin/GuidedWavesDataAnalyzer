import datetime
import threading
from os import path
from threading import Thread
from threading import Event
import PySimpleGUI as sg
from matplotlib import pyplot as pp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import glob
import csv
from ZoomPan import ZoomPan
from Frame import *
from Snapshot import Snapshot
from NVSerial import *


class GUI:
    def __init__(self):
        self._VARS = {'window': False,
                      'fig_agg': False,
                      'pltFig': False,
                      'axFig': False,
                      'values': [],
                      'fig_index': 0}
        self._PLOT_PARAM = {'title': False,
                            'x_label': False,
                            'y_label': False,
                            'grid': True,
                            'y_min': False,
                            'y_max': False}
        self.data = []
        self.frames = []
        self.c_mean = 0
        self.func = None
        self.styles = {'first data': 'k-'}

        sg.theme('LightGray1')

        param_column = [[sg.Text("Guide Length", key='word1', size=(12, 0)),
                         sg.Input("1.0", size=(10, 0), key='gLenInput', enable_events=True),
                         sg.Text("m", key='len_u', size=(3, 0)),
                         sg.Button('Set', key='set_btn')],
                        [sg.Text("Zo", key='word2', size=(12, 0)),
                         sg.Input("75", size=(10, 0), key='zoInput', enable_events=True),
                         sg.Text("\u03A9", key='zo_u', size=(6, 0))],
                        [sg.Text("Capacitance", key='word3', size=(12, 0)),
                         sg.InputText('',
                                      size=(10, 0),
                                      text_color='black',
                                      use_readonly_for_disable=True,
                                      disabled=True,
                                      key='CapOutput'),
                         sg.Text("F", key='C_u', size=(6, 0))],
                        [sg.Text("Inductance", key='word4', size=(12, 0)),
                         sg.InputText('',
                                      size=(10, 0),
                                      text_color='black',
                                      use_readonly_for_disable=True,
                                      disabled=True,
                                      key='IndOutput'),
                         sg.Text("H", key='L_u', size=(6, 0))],
                        [sg.Text("wave speed", key='word4', size=(12, 0)),
                         sg.InputText('',
                                      size=(10, 0),
                                      text_color='black',
                                      use_readonly_for_disable=True,
                                      disabled=True,
                                      key='WSOutput'),
                         sg.Text("m/s", key='ws_u', size=(6, 0))],
                        [sg.Text("\u03B5r", key='word4', size=(12, 0)),
                         sg.InputText('',
                                      size=(10, 0),
                                      text_color='black',
                                      use_readonly_for_disable=True,
                                      disabled=True,
                                      key='RPOutput')]]

        layout = [
            [sg.Button('Connect', key='connect_serial'),
             sg.Button('Plot Ref', key='ref_btn', visible=False),
             sg.Button('Test', key='test_btn', visible=False)],
            [sg.Canvas(key='canvas', size=(300, 400)),
             sg.Frame(layout=param_column, title="frame", key='param_col')
             ],
            [sg.Button('Plot', key='plot_bt'), sg.Quit('Exit')],
        ]

        self._VARS['window'] = sg.Window('GUIded Waves',
                                         layout,
                                         finalize=True,
                                         return_keyboard_events=True)

    def start(self):
        self.draw_chart()
        names_arr = []
        data_to_plot = []
        zp = ZoomPan()

        self.update_chart()
        tst = False
        while True:
            event, self._VARS['values'] = self._VARS['window'].Read()

            if event in (None, 'Sair'):
                break
            elif event == 'connect_serial':
                try:
                    nanovna = NVSerial()
                    nanovna.open()
                    nanovna.set_frequencies(1e6, 1200e6, 101)
                    self._VARS['window']['ref_btn'].Update(visible=True)
                    self._VARS['window']['test_btn'].Update(visible=True)
                    self._VARS['window']['connect_serial'].Update("Disconnect")
                    self.clear_chart()
                    self.update_chart()
                except:
                    nanovna.close()
                    self._VARS['window']['ref_btn'].Update(visible=False)
                    self._VARS['window']['test_btn'].Update(visible=False)
                    self._VARS['window']['connect_serial'].Update("Connect")
                # print('already connected')
            elif event == 'ref_btn':
                gLen = float(self._VARS['values']['gLenInput'])
                Zo = float(self._VARS['values']['zoInput'])
                C = (0, " \u03BCF")
                L = (0, "mH")
                Er = 0
                WS = 0
                frame = set_frame(nanovna, Zo, gLen)

                C, L, WS, Er = calculate_params(Zo, frame.cable.t_break, gLen)

                C = set_unit_prefix(C, "F")
                L = set_unit_prefix(L, "H")

                self.update_param_frame((C, L, Er, WS))

                self.clear_chart()

                frame.update(C, L, WS, Er)

                # frame.plot(self._VARS['pltFig'])

                self.frames.append(frame)

                plot_frames(self.frames,self._VARS['pltFig'])

                self.update_chart()
                figZoom = zp.zoom_factory(base_scale=1.1)
                figPan = zp.pan_factory(plt.gca())
            elif event == 'test_btn':
                tst = tst ^ True
                gLen = float(self._VARS['values']['gLenInput'])
                Zo = float(self._VARS['values']['zoInput'])
                if tst is True:

                    ev = Event()
                    self._VARS['window']['test_btn'].Update("Stop Test")
                    t = Thread(target=test, args=(self.frames, nanovna, Zo, gLen, ev, 3))
                    t.start()
                    # t.join()
                else:
                    self._VARS['window']['test_btn'].Update("Test")
                    ev.set()
                    t.join()

                    f0 = self.frames[0]

                    C, L, WS, Er = calculate_params(float(Zo), f0.cable.t_break, float(gLen))

                    for frame in self.frames:
                        frame.update(C, L, WS, Er)

                    self.clear_chart()
                    p = 'D:\\git\\GuidedWavesDataAnalyzer\\out.csv'
                    save_frame_collection_as_csv(self.frames, p)
                    plot_frames(self.frames, self._VARS['pltFig'], q=10)

                    self.update_chart()
                figZoom = zp.zoom_factory(base_scale=1.1)
                figPan = zp.pan_factory(plt.gca())
            elif event == 'set_btn':

                glen = validate_field_value(self._VARS['values']['gLenInput'])
                self._VARS['window']['gLenInput'].Update(value=glen)
                zo = validate_field_value(self._VARS['values']['zoInput'])
                self._VARS['window']['zoInput'].Update(value=zo)

                if len(self.frames) > 0:
                    f0 = self.frames[0]

                    C, L, WS, Er = calculate_params(float(zo), f0.cable.t_break, float(glen))

                    C = set_unit_prefix(C, "F")
                    L = set_unit_prefix(L, "H")

                else:
                    C = (0, " \u03BCF")
                    L = (0, "mH")
                    Er = 0
                    WS = 0

                self.update_param_frame((C, L, Er, WS))

                self.clear_chart()
                if len(self.frames) > 0:
                    for frame in self.frames:
                        frame.update(C, L, WS, Er)

                    plot_frames(self.frames, self._VARS['pltFig'])

                self.update_chart()
                figZoom = zp.zoom_factory(base_scale=1.1)
                figPan = zp.pan_factory(plt.gca())
            elif event == 'plot_bt':

                gLen = float(self._VARS['values']['gLenInput'])
                Zo = float(self._VARS['values']['zoInput'])

                frame = set_frame(nanovna, Zo, gLen)

                self.clear_chart()

                frame.update(C, L, WS, Er)

                self.frames.append(frame)

                plot_frames(self.frames, self._VARS['pltFig'])
                # for f in self.frames:
                # f.plot(self._VARS['pltFig'])

                self.update_chart()

                figZoom = zp.zoom_factory(base_scale=1.1)
                figPan = zp.pan_factory(plt.gca())
            elif event == 'MouseWheel:Up':
                pass
            elif event == 'MouseWheel:Down':
                pass
            else:
                pass

        self._VARS['window'].close()

    def draw_figure(self, canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    def draw_chart(self, **styles):
        self._VARS['pltFig'] = pp.figure()
        # self._VARS['axFig'] = self._VARS['pltFig'].add_subplot(1, 1, 1)
        self._prepare_plot(**styles)
        self._VARS['fig_agg'] = self.draw_figure(self._VARS['window']['canvas'].TKCanvas, self._VARS['pltFig'])

    def update_chart(self, **styles):
        self._VARS['fig_agg'].get_tk_widget().forget()
        # pp.clf()
        self._prepare_plot(**styles)
        self._VARS['fig_agg'] = self.draw_figure(self._VARS['window']['canvas'].TKCanvas, self._VARS['pltFig'])

    def clear_chart(self):
        pp.clf()

    def _prepare_plot(self, **styles):
        if len(self.data) == 0:
            return

        for d in self.data:
            if d[2] == 'original data':
                pp.plot(d[0], d[1], label=d[2])
                self._PLOT_PARAM['y_max'] = 1.1 * np.max(d[1])
                self._PLOT_PARAM['y_min'] = np.min(d[1]) - 0.01 * np.max(d[1])
            else:
                pp.plot(d[0], d[1], label=d[2])

        pp.grid(self._PLOT_PARAM['grid'])
        pp.title(self._PLOT_PARAM['title'])
        pp.xlabel(self._PLOT_PARAM['x_label'])
        pp.ylabel(self._PLOT_PARAM['y_label'])
        pp.ylim(ymax=self._PLOT_PARAM['y_max'], ymin=self._PLOT_PARAM['y_min'])
        self._PLOT_PARAM['y_label']
        pp.legend(loc='upper left', prop={'size': 6})

    def get_data(self):
        p = self._VARS['values']['FolderSelected']
        files = glob.glob(glob.escape(p) + "/*.csv")

        print(files)
        for file in files:
            if path.isfile(file):
                return self.__load_csv(file)
            else:
                raise ValueError("O CAMINHO É INVÁLIDO")

    def __load_csv(self, csv_path):
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            t = []
            s = []

            for row in csv_reader:
                if len(row) < 3 or row[0] == "Time":
                    pass
                else:
                    t.append(float(row[0]))
                    s.append(float(row[1]))

        # x = np.array(t)
        # y = np.array(s)
        return [[t, s, 'original data']]

    def update_param_frame(self, data):
        C, L, Er, WS = data

        self._VARS['window']['CapOutput'].Update(value=f'{C[0]:.2f}')
        self._VARS['window']['C_u'].Update(value=C[1])
        self._VARS['window']['IndOutput'].Update(value=f'{L[0]:.2f}')
        self._VARS['window']['L_u'].Update(value=L[1])
        self._VARS['window']['RPOutput'].Update(value=f'{Er:.2f}')
        self._VARS['window']['WSOutput'].Update(value=int(WS))



def validate_field_value(newval: str):
    try:
        nv = float(newval)
    except:
        nv = 1.0
    return nv

def validate_field_value_old(newval: str):
    alphalist = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    alphalist += [chr(i) for i in range(ord('a'), ord('z') + 1)]

    if newval == '' or newval == '-':
        return 0
    if newval.__contains__('-'):
        return newval[:-1]
    if any(x in newval for x in alphalist):
        if len(newval) >= 2:
            return newval[:-1]
        else:
            return 0
    if len(newval) > 1 and newval.startswith('0') and not (newval.__contains__('.') or newval.__contains__(',')):
        return newval[-1]
    if (newval.__contains__('.') or newval.__contains__(',')) and len(newval) == 1:
        return '0.'
    if newval.count('.') > 1:
        if len(newval) >= 2:
            return newval[:-1]
        else:
            return 0
    else:
        return newval


def set_frame(nanovna: NVSerial, zo, l, spf=10, delay=0.1):
    from datetime import datetime
    SNAPSHOTS_PER_FRAME = spf
    ss_list = []
    for i in range(0, SNAPSHOTS_PER_FRAME):
        data = nanovna.scan()
        now = datetime.now()  # current date and time
        name = datetime.timestamp(now)
        ss = Snapshot(data, name)
        ss_list.append(ss)
        time.sleep(delay)

    f = Frame(zo, l)
    f.add_snapshots_list(ss_list)
    f.run()

    return f


def calculate_params(zo, t_break, length):
    if not t_break > 0:
        raise ValueError("t_break must be greater than 0")

    speed = (2 * length) / t_break

    c = 1 / (zo * speed)

    l = 1 / (c * speed ** 2)

    e0 = 8.85e-12
    u0 = 4 * np.pi * 1e-7

    rp = 1 / (speed ** 2 * e0 * u0)

    return c, l, speed, rp


def test(frames, nanovna, Zo, gLen, event, delay):
    i = 0
    while True:
        if event.is_set():
            break
        frame = set_frame(nanovna, Zo, gLen, spf=1, delay=delay)
        frames.append(frame)
        i += 1
        print("frames taken: " + str(i))


def save_frame_collection_as_csv(frames, p):
    df = DataFrame()

    for frame in frames:
        df[frame.name] = frame.average[0]

    from pathlib import Path

    filepath = Path(p)

    # filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)


tela = GUI()
tela.start()
