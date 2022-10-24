from os import path
import PySimpleGUI as sg
import numpy as np
from matplotlib import pyplot as pp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import glob
import csv
import Plots

class GUI:
    def __init__(self):
        self._VARS = {'window': False,
                      'fig_agg': False,
                      'pltFig': False,
                      'values': []}
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

        param_column = [[sg.Text("blablabla", key='word1', size=(12, 0))],
                        [sg.Text("trololo", key='word2', size=(12, 0))],
                        [sg.Text("pipipi", key='word3', size=(12, 0))],
                        [sg.Text("popopo", key='word4', size=(12, 0))]]

        layout = [
            [sg.Text('Root Folder:', size=(9, 0)),
             sg.In(size=(25, 1), enable_events=True, key='FolderSelected'),
             sg.FolderBrowse(button_text='Procurar', key='folder_browser')],
            [sg.Button('<<', key='backward'),
             sg.Canvas(key='canvas', size=(300, 400)),
             sg.Button('>>', key='forward'),
             sg.Frame(layout=param_column, title="frame", key='param_col')
             ],
            [sg.Button('Executar'), sg.Quit('Sair')],
        ]

        self._VARS['window'] = sg.Window('Guided Waves TDR Data Analyzer', layout, finalize=True)

    def start(self):
        self.draw_chart()
        while True:
            event, self._VARS['values'] = self._VARS['window'].Read()

            if event in (None, 'Sair'):
                break
            elif event == 'FolderSelected':
                p = self._VARS['values']['FolderSelected']
                files = glob.glob(glob.escape(p) + "/*.csv")

                print(files)
                for i in range(0, len(files)):


                    Plots.import_data_and_print("INITIAL", files[i], self._VARS['window']['canvas'].TKCanvas)

                # self.data = self.get_data()
                self.update_chart()
                pass
            elif event == 'backward':
                self._VARS['window']['word1'].Update("backward")
                pass
            elif event == 'forward':
                self._VARS['window']['word1'].Update("forward")
                pass
            else:
                self.update_chart()

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
        pp.clf()
        self._prepare_plot(**styles)
        self._VARS['fig_agg'] = self.draw_figure(self._VARS['window']['canvas'].TKCanvas, self._VARS['pltFig'])

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

    def time_formatter(self, t):
        stime = ''
        t_m = t // 60
        t_h = t_m // 60
        t_s = t % 60
        if t_h > 0 or t_m > 0:
            if t_h > 0:
                stime = "{t}".format(t=int(t_h)) + ' hora'
                if t_h > 1:
                    stime += 's'
                stime += ', '
            if t_m > 0:
                stime = "{t}".format(t=int(t_m)) + ' minuto'
                if t_m > 1:
                    stime += 's'
                stime += ', '
            if t_s > 0:
                stime = "{t}".format(t=int(t_s)) + ' segundo'
                if t_s > 1:
                    stime += 's'
                stime += ', '
        else:
            if t_s >= 1:
                stime = "{t:.2f}".format(t=t_s) + ' segundo'
                if t_s >= 2:
                    stime += 's'
                stime += ', '
            elif not t_s - int(t_s) == 0:
                ok = False
                t_s1 = t_s
                idx = 0
                while not ok:
                    t_s1 = t_s1 * 10
                    idx += 1
                    if t_s1 >= 1:
                        ok = True
                if not idx % 3 == 0:
                    while not idx % 3 == 0:
                        idx += 1
                        t_s1 *= 10
                stime += "{t:.2f}".format(t=t_s1)
                if idx / 3 == 1:
                    stime += ' ms'
                elif idx / 3 == 2:
                    stime += ' µs'
                elif idx / 3 == 3:
                    stime += ' ns'
                elif idx / 3 == 4:
                    stime += ' ps'
                elif idx / 3 == 5:
                    stime += ' fs'
                elif idx / 3 == 6:
                    stime += ' as'
                elif idx / 3 == 7:
                    stime += ' zs'
                elif idx / 3 == 8:
                    stime += ' ys'
        return stime


tela = GUI()
tela.start()
