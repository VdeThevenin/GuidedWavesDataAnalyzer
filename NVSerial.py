import time
import serial
import numpy as np
from serial.tools import list_ports

VIDPIDs = set([(0x0483, 0x5740), (0x04b4, 0x0008)])
REF_LEVEL = (1<<9)
class NVSerial:
    def __init__(self, dev=None):
        self.dev = dev or getport()
        self.serial = None
        self._frequencies = None
        self.points = 101

    @property
    def frequencies(self):
        return self._frequencies

    def set_frequencies(self, start=5e3, stop=1200e6, points=None):
        if points:
            self.points = points
        self._frequencies = np.linspace(start, stop, self.points)

    def open(self):
        if self.serial is None:
            self.serial = serial.Serial(self.dev)

    def close(self):
        if self.serial:
            self.serial.close()
        self.serial = None

    def send_command(self, cmd):
        self.open()
        self.serial.write(cmd.encode())
        self.serial.readline()  # discard empty line

    def set_sweep(self, start, stop):
        if start is not None:
            self.send_command("sweep start %d\r" % start)
        if stop is not None:
            self.send_command("sweep stop %d\r" % stop)

    def set_frequency(self, freq):
        if freq is not None:
            self.send_command("freq %d\r" % freq)

    def set_port(self, port):
        if port is not None:
            self.send_command("port %d\r" % port)

    def set_gain(self, gain):
        if gain is not None:
            self.send_command("gain %d %d\r" % (gain, gain))

    def set_offset(self, offset):
        if offset is not None:
            self.send_command("offset %d\r" % offset)

    def set_strength(self, strength):
        if strength is not None:
            self.send_command("power %d\r" % strength)

    def set_filter(self, filter):
        self.filter = filter

    def fetch_data(self):
        result = ''
        line = ''
        while True:
            c = self.serial.read().decode('utf-8')
            if c == chr(13):
                next  # ignore CR
            line += c
            if c == chr(10):
                result += line
                line = ''
                next
            if line.endswith('ch>'):
                # stop on prompt
                break
        return result

    def fetch_buffer(self, freq=None, buffer=0):
        self.send_command("dump %d\r" % buffer)
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.extend([int(d, 16) for d in line.strip().split(' ')])
        return np.array(x, dtype=np.int16)

    def fetch_rawwave(self, freq=None):
        if freq:
            self.set_frequency(freq)
            time.sleep(0.05)
        self.send_command("dump 0\r")
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.extend([int(d, 16) for d in line.strip().split(' ')])
        return np.array(x[0::2], dtype=np.int16), np.array(x[1::2], dtype=np.int16)

    def fetch_array(self, sel):
        self.send_command("data %d\r" % sel)
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.extend([float(d) for d in line.strip().split(' ')])
        return np.array(x[0::2]) + np.array(x[1::2]) * 1j

    def fetch_gamma(self, freq=None):
        if freq:
            self.set_frequency(freq)
        self.send_command("gamma\r")
        data = self.serial.readline()
        d = data.strip().split(' ')
        return (int(d[0]) + int(d[1]) * 1.j) / REF_LEVEL

    def resume(self):
        self.send_command("resume\r")

    def pause(self):
        self.send_command("pause\r")

    def scan_gamma0(self, port=None):
        self.set_port(port)
        return np.vectorize(self.gamma)(self.frequencies)

    def scan_gamma(self, port=None):
        self.set_port(port)
        return np.vectorize(self.fetch_gamma)(self.frequencies)

    def data(self, array=0):
        self.send_command("data %d\r" % array)
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                d = line.strip().split(' ')
                x.append(float(d[0]) + float(d[1]) * 1.j)
        return np.array(x)

    def fetch_frequencies(self):
        self.send_command("frequencies\r")
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.append(float(line))
        self._frequencies = np.array(x)

    def send_scan(self, start=1e6, stop=900e6, points=None):
        if points:
            self.send_command("scan %d %d %d\r" % (start, stop, points))
        else:
            self.send_command("scan %d %d\r" % (start, stop))

    def scan(self):
        segment_length = self.points
        array0 = []
        array1 = []
        if self._frequencies is None:
            self.fetch_frequencies()
        freqs = self._frequencies
        while len(freqs) > 0:
            seg_start = freqs[0]
            seg_stop = freqs[segment_length - 1] if len(freqs) >= segment_length else freqs[-1]
            length = segment_length if len(freqs) >= segment_length else len(freqs)
            # print((seg_start, seg_stop, length))
            self.send_scan(seg_start, seg_stop, length)
            array0.extend(self.data(0))
            array1.extend(self.data(1))
            freqs = freqs[segment_length:]
        self.resume()
        return (array0, array1, self.frequencies)

    def skrf_network(self, x):
        import skrf as sk
        n = sk.Network()
        n.frequency = sk.Frequency.from_f(self.frequencies / 1e6, unit='mhz')
        n.s = x
        return n

    def smith(self, x):
        n = self.skrf_network(x)
        n.plot_s_smith()
        return n


def getport() -> str:
    device_list = list_ports.comports()
    for device in device_list:
        if (device.vid, device.pid) in VIDPIDs:
            return device.device
    raise OSError("device not found")