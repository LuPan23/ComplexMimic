import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_color_gradient(percent, color='Blues'):
    return mpl.colormaps[color](percent)[:3]

def agt_color(aidx):
    return matplotlib.colors.to_rgb(plt.rcParams['axes.prop_cycle'].by_key()['color'][aidx % 10])