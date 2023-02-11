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
                      'fig_agg': None,
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
                         sg.Input("1.0", size=(10, 0), key='glenInput', enable_events=True),
                         sg.Text("m", key='len_u', size=(3, 0)),
                         sg.Button('Set', key='set_btn')],
                        [sg.Text("zo", key='word2', size=(12, 0)),
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
             sg.Button('Single', key='pn_btn', visible=False),
             sg.Button('Periodic', key='capture_btn', visible=False),
             sg.Button('Real Time', key='rt_btn', visible=False)],
            [sg.Canvas(key='canvas', size=(300, 400)),
             sg.Frame(layout=param_column, title="frame", key='param_col')
             ],
            [sg.Quit('Exit')],
        ]

        self._VARS['window'] = sg.Window('GUIded Waves',
                                         layout,
                                         finalize=True,
                                         return_keyboard_events=True)

    def start(self):
        self.draw_chart()
        zp = ZoomPan()

        self.update_chart()
        capt = False
        rt = False
        while True:
            event, self._VARS['values'] = self._VARS['window'].Read(timeout=200)

            glen = validate_field_value(self._VARS['values']['glenInput'])
            zo = validate_field_value(self._VARS['values']['zoInput'])

            if event in (None, 'Exit'):
                break
            elif event == 'connect_serial':
                try:
                    nanovna = NVSerial()
                except OSError:
                    nanovna = None
                    continue
                try:
                    nanovna.open()
                    nanovna.set_frequencies(1e6, 1200e6, 101)
                    self._VARS['window']['ref_btn'].Update(visible=True)
                    self._VARS['window']['pn_btn'].Update(visible=True)
                    self._VARS['window']['capture_btn'].Update(visible=True)
                    self._VARS['window']['rt_btn'].Update(visible=True)
                    self._VARS['window']['connect_serial'].Update("Disconnect")
                    clear_chart()
                    self.update_chart()
                except:
                    nanovna.close()
                    self._VARS['window']['ref_btn'].Update(visible=False)
                    self._VARS['window']['pn_btn'].Update(visible=False)
                    self._VARS['window']['capture_btn'].Update(visible=False)
                    self._VARS['window']['rt_btn'].Update(visible=False)
                    self._VARS['window']['connect_serial'].Update("Connect")
                # print('already connected')
            elif event == 'pn_btn':
                frame = set_frame(nanovna, zo, glen)

                c, ind, ws, er = calculate_params(zo, frame.cable.t_break, glen)
                c = set_unit_prefix(c, "F")
                ind = set_unit_prefix(ind, "H")
                self.update_param_frame((c, ind, er, ws))

                clear_chart()
                if len(self.frames) == 0:
                    frame.update(c, ind, ws, er)
                else:
                    frame.update(c, ind, ws, er, self.frames[0].ibreak)
                self.frames.append(frame)
                plot_frames(self.frames, self._VARS['pltFig'])
                self.update_chart()

                zp.zoom_factory(base_scale=1.1)
                zp.pan_factory(plt.gca())
            elif event == 'ref_btn':
                frame = set_frame(nanovna, zo, glen)

                c, ind, ws, er = calculate_params(zo, frame.cable.t_break, glen)
                c = set_unit_prefix(c, "F")
                ind = set_unit_prefix(ind, "H")
                self.update_param_frame((c, ind, er, ws))

                clear_chart()
                if len(self.frames) == 0:
                    frame.update(c, ind, ws, er)
                else:
                    frame.update(c, ind, ws, er, self.frames[0].ibreak)
                self.frames = [frame]
                plot_frames(self.frames, self._VARS['pltFig'])
                self.update_chart()

                zp.zoom_factory(base_scale=1.1)
                zp.pan_factory(plt.gca())
            elif event == 'capture_btn':
                capt = capt ^ True

                if capt is True:
                    self._VARS['window']['capture_btn'].Update("Stop Periodic")
                    ev = Event()
                    t = Thread(target=capture, args=(self.frames, nanovna, zo, glen, ev, 0.1))
                    t.start()
                else:
                    self._VARS['window']['capture_btn'].Update("Periodic")
                    ev.set()
                    t.join(timeout=1)

                    f0 = self.frames[0]

                    c, ind, ws, er = calculate_params(float(zo), f0.cable.t_break, float(glen))

                    for frame in self.frames:
                        frame.update(c, ind, ws, er, self.frames[0].ibreak)

                    clear_chart()
                    p = './out.csv'
                    save_frame_collection_as_csv(self.frames, p)
                    plot_frames(self.frames, self._VARS['pltFig'], q=10)

                    self.update_chart()
                zp.zoom_factory(base_scale=1.1)
                zp.pan_factory(plt.gca())

            elif event == 'rt_btn':

                frame = set_frame(nanovna, zo, glen, spf=1, delay=0)
                c, ind, ws, er = calculate_params(zo, frame.cable.t_break, glen)
                frame.update(c, ind, ws, er)
                self.frames.append(frame)
                self.frames.append(frame)
                rt = rt ^ True

                if rt is True:
                    self._VARS['window']['rt_btn'].Update("Stop Real Time")
                else:
                    self._VARS['window']['rt_btn'].Update("Real Time")

                zp.zoom_factory(base_scale=1.1)
                zp.pan_factory(plt.gca())
            elif event == 'set_btn':

                self._VARS['window']['glenInput'].Update(value=glen)
                self._VARS['window']['zoInput'].Update(value=zo)

                if len(self.frames) > 0:
                    f0 = self.frames[0]

                    c, ind, ws, er = calculate_params(float(zo), f0.cable.t_break, float(glen))

                    c = set_unit_prefix(c, "F")
                    ind = set_unit_prefix(ind, "H")

                else:
                    c = (0, " \u03BCF")
                    ind = (0, "mH")
                    er = 0
                    ws = 0

                self.update_param_frame((c, ind, er, ws))

                clear_chart()
                if len(self.frames) > 0:
                    for frame in self.frames:
                        frame.update(c, ind, ws, er, self.frames[0].ibreak)

                    plot_frames(self.frames, self._VARS['pltFig'])

                self.update_chart()
                zp.zoom_factory(base_scale=1.1)
                zp.pan_factory(plt.gca())
            elif event == 'MouseWheel:Up':
                pass
            elif event == 'MouseWheel:Down':
                pass
            else:
                pass

            if rt is True:
                clear_chart()
                frame = set_frame(nanovna, zo, glen, spf=1, delay=0)
                c, ind, ws, er = calculate_params(zo, frame.cable.t_break, glen)
                frame.update(c, ind, ws, er, self.frames[0].ibreak)
                self.frames[-1] = frame
                plot_frames(self.frames, self._VARS['pltFig'])
                self.update_chart()

        self._VARS['window'].close()

    def draw_chart(self, **styles):
        self._VARS['pltFig'] = pp.figure()
        self._VARS['fig_agg'] = draw_figure(self._VARS['window']['canvas'].TKCanvas, self._VARS['pltFig'])

    def update_chart(self, **styles):
        self._VARS['fig_agg'].get_tk_widget().forget()
        self._VARS['fig_agg'] = draw_figure(self._VARS['window']['canvas'].TKCanvas, self._VARS['pltFig'])

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
        c, ind, er, ws = data

        self._VARS['window']['CapOutput'].Update(value=f'{c[0]:.2f}')
        self._VARS['window']['C_u'].Update(value=c[1])
        self._VARS['window']['IndOutput'].Update(value=f'{ind[0]:.2f}')
        self._VARS['window']['L_u'].Update(value=ind[1])
        self._VARS['window']['RPOutput'].Update(value=f'{er:.2f}')
        self._VARS['window']['WSOutput'].Update(value=int(ws))


def clear_chart():
    pp.clf()


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def validate_field_value(newval: str):
    try:
        nv = float(newval)
    except:
        nv = 1.0
    return nv


def set_frame(nanovna: NVSerial, zo, ind, spf=10, delay=0.1):
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

    f = Frame(zo, ind)
    f.add_snapshots_list(ss_list)
    f.run()

    return f


def calculate_params(zo, t_break, length):
    if not t_break > 0:
        raise ValueError("t_break must be greater than 0")

    speed = (2 * length) / t_break

    c = 1 / (zo * speed)

    ind = 1 / (c * speed ** 2)

    e0 = 8.85e-12
    u0 = 4 * np.pi * 1e-7

    rp = 1 / (speed ** 2 * e0 * u0)

    return c, ind, speed, rp


def capture(frames, nanovna, zo, glen, event, delay):
    i = 0
    while True:
        if event.is_set():
            break
        frame = set_frame(nanovna, zo, glen, spf=10, delay=delay)
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
