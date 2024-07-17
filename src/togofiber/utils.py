import math


def print_duration(d: float, desc=None):
    d = math.ceil(d)
    desc = desc or "DURATION"
    print(f"{desc}: {d//60:02d} mins {d%60:02d} s")
