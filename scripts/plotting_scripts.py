# Custom adapted function for displaying values on bar graphs
# Original credit to Secant Zhang, Jun 26 2019
# "Seaborn Barplot - Displaying Values", StackOverflow
# URL: https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
import numpy as np
import matplotlib.patheffects as PathEffects

def show_values_on_bars(axs, h_v="v", fontsize = 12, space=0.1):
    space = float(space)
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + space
                value = round(p.get_height(), 2)

                if value > 0: ax.text(_x, _y, value, ha="center", fontsize = fontsize, alpha = 1) 
                else: ax.text(_x, _y - (3*space), value, ha="center", fontsize = fontsize, alpha = 1) 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + space
                _y = p.get_y() + p.get_height()
                value = round(p.get_width(), 2)
                
                if value > 0: ax.text(_x, _y, value, ha="left", fontsize = fontsize, alpha = 1)
                else: ax.text(_x - (2*space), _y, value, ha="right", fontsize = fontsize, alpha = 1)
    
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
