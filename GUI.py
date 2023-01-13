from os import path
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as pp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import glob
import csv
import DataPlotter
from DataPlotter import Data
from ZoomPan import ZoomPan
# import Plots


class GUI:
    def __init__(self):
        self._VARS = {'window': False,
                      'fig_agg': False,
                      'pltFig': False,
                      'values': [],
                      'fig_index': 0}
        self._PLOT_PARAM = {'title': False,
                            'x_label': False,
                            'y_label': False,
                            'grid': True,
                            'y_min': False,
                            'y_max': False}
        self.data = []
        self.c_mean = 0
        self.func = None
        self.styles = {'first data': 'k-'}

        sg.theme('LightGray1')

        param_column = [[sg.Text("Guide Length", key='word1', size=(12, 0)),
                         sg.Input("1.0", size=(10, 0), key='gLenInput', enable_events=True),
                         sg.Text("m", key='len_u', size=(3, 0))],
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
        folder_frame = []

        layout = [
            [sg.Text('Root Folder:', size=(15, 0)),
             sg.In(size=(80, 1), enable_events=True, key='FolderSelected'),
             sg.FolderBrowse(button_text='Browse', key='folder_browser')],
            [sg.Listbox([], size=(30, 30), select_mode='extended', key='ListBox'),
             sg.Canvas(key='canvas', size=(300, 400)),
             sg.Frame(layout=param_column, title="frame", key='param_col')
             ],
            [sg.Button('Plot', key='plot_bt'), sg.Quit('Sair')],
        ]

        self._VARS['window'] = sg.Window('GUIded Waves TDR Data Analyzer', layout, finalize=True, return_keyboard_events=True)

    def start(self):
        self.draw_chart()
        names_arr = []
        data_arr = []
        selected = []
        data_to_plot = []
        zp = ZoomPan()

        while True:
            event, self._VARS['values'] = self._VARS['window'].Read()

            if event in (None, 'Sair'):
                break
            elif event == 'FolderSelected':
                names_arr = []
                p = self._VARS['values']['FolderSelected']

                types = ('/*.csv', '/*.s1p')  # the tuple of file types
                files = []
                for tp in types:
                    files.extend(glob.glob(glob.escape(p) + tp))

                # initial_data = Data(f, "INITIAL",2.0)

                for file in files:
                    names_arr.append(file[file.rfind('\\')+1:])

                self._VARS['window']['ListBox'].Update(values=names_arr)

                # DataPlotter.plot_data(initial_data, data_arr[0], self._VARS['pltFig'])
                # Plots.import_data_and_print("INITIAL", files[self._VARS['fig_index']], self._VARS['pltFig'])
                # self.update_chart()
                pass
            elif event == 'gLenInput' or event == 'zoInput':

                glen = validate_field_value(self._VARS['values']['gLenInput'])
                self._VARS['window']['gLenInput'].Update(value=glen)
                zo = validate_field_value(self._VARS['values']['zoInput'])
                self._VARS['window']['zoInput'].Update(value=zo)

                Er = 0

                if len(data_to_plot) != 0:
                    for data in data_to_plot:
                        data.set_GuideLen(float(glen))
                        data.set_Zo(float(zo))
                        if data_to_plot.index(data) == 0:
                            C = DataPlotter.set_unit_prefix(data.capacitance, "F")
                            L = DataPlotter.set_unit_prefix(data.inductance, "H")
                            Er = data.relative_permissivity
                            WS = data.speed
                            self.update_param_frame((C, L, Er, WS))

                    if Er == 0:
                        C = DataPlotter.set_unit_prefix(data_to_plot[0].capacitance, "F")
                        L = DataPlotter.set_unit_prefix(data_to_plot[0].inductance, "H")
                        Er = data_to_plot[0].relative_permissivity
                        WS = data_to_plot[0].speed
                        self.update_param_frame((C, L, Er, WS))

            elif event == 'plot_bt':
                data_to_plot = []
                selected = self._VARS['window']['ListBox'].get_indexes()

                gLen = float(self._VARS['values']['gLenInput'])
                Zo = float(self._VARS['values']['zoInput'])
                C = (0, " \u03BCF")
                L = (0, "mH")
                Er = 0
                WS = 0

                for i in selected:
                    s1p = False
                    if files[i].__contains__(".s1p") or files[i].__contains__(".S1P"):
                        s1p = True

                    data_to_plot.append(Data(files[i], names_arr[i], gLen, Zo, s1p))
                    if names_arr[i].__contains__("ORIG"):
                        C = DataPlotter.set_unit_prefix(data_to_plot[-1].capacitance, "F")
                        L = DataPlotter.set_unit_prefix(data_to_plot[-1].inductance, "H")
                        Er = data_to_plot[-1].relative_permissivity
                        WS = data_to_plot[-1].speed

                if Er == 0:
                    C = DataPlotter.set_unit_prefix(data_to_plot[0].capacitance, "F")
                    L = DataPlotter.set_unit_prefix(data_to_plot[0].inductance, "H")
                    Er = data_to_plot[0].relative_permissivity
                    WS = data_to_plot[0].speed

                    DataPlotter.get_statistics(data_to_plot)

                self.update_param_frame((C, L, Er, WS))

                self.clear_chart()

                DataPlotter.plot_array(data_to_plot, "teste.py", self._VARS['pltFig'])

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
                self._PLOT_PARAM['y_max'] = 1.1*np.max(d[1])
                self._PLOT_PARAM['y_min'] = np.min(d[1]) - 0.01*np.max(d[1])
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
        files = glob.glob(glob.escape(p)+ "/*.csv")

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


def validate_field_value(newval : str):
    alphalist = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    alphalist += [chr(i) for i in range(ord('a'), ord('z')+1)]

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

tela = GUI()
tela.start()
