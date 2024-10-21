import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.rcParams.update({'font.size': 12})


def plot_lines(data, label_x='', label_y='', title='', l_width=2, lim=None):
    fig, ax = plt.subplots(1, 1)
    for method_name, (x, y) in data:
        ax.plot(x, y, linewidth=l_width, label=method_name)
    if lim is not None:
        m_style = {
            'marker': 'o',
            'size': 5,
            'edge_color': 'black',
            'edge_width': 1,
            'face_color': 'white'
        }
        ax_ins = ax.inset_axes((0.2, 0.4, 0.4, 0.5))
        for _, (x, y) in data:
            ax_ins.plot(x, y, linewidth=l_width, marker=m_style['marker'], markersize=m_style['size'],
                        markeredgecolor=m_style['edge_color'], markerfacecolor=m_style['face_color'])
        if lim is None:
            raise ValueError('lim space not set')
        x_lim0, x_lim1, y_lim0, y_lim1 = lim
        ax_ins.set_xlim(x_lim0, x_lim1)
        ax_ins.set_ylim(y_lim0, y_lim1)
        mark_inset(ax, ax_ins, loc1=2, loc2=4, fc="none", ec='k', lw=1, linestyle='-.')
    plt.legend()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.show()