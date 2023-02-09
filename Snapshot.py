from ifft import ifft


class Snapshot:
    def __init__(self, s1p, name):
        self.name = name
        self.s1p = s1p
        self.t, self.s11, self.z, self.td_w_offset, self.td_wo_offset = ifft(s1p, nfft=1001)

    def get_time(self):
        return self.t

    def get_s11(self):
        return self.s11

    def get_z_response(self):
        return self.z

    def get_name(self):
        return self.name
