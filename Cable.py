import numpy as np


class Cable:
    def __init__(self, zo, length):
        if not zo > 0:
            raise ValueError("Zo must be greater than 0")
        if not length > 0:
            raise ValueError("Cable Length must be greater than 0")

        self.zo = zo
        self.length = length
        self.t_break = 0
        self.v_break = 0

        self.speed = 0
        self.capacitance = 0
        self.inductance = 0
        self.relative_permissivity = 0

    def set_tbreak(self, tb):
        self.t_break = tb

    def set_vbreak(self, vb):
        self.v_break = vb
