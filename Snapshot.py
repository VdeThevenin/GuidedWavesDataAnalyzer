from ifft import ifft


class Snapshot:
    def __init__(self, s1p, name):
        self.name = name
        self.t, self.s11, self.z = ifft(s1p, nfft=1001)

    def get_time(self):
        return self.t

    def get_s11(self):
        return self.s11

    def get_z_response(self):
        return self.z

    def get_name(self):
        return self.name
