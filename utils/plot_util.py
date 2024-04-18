import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})


def plot_lines(data, label_x='', label_y='', title='', l_width=2):
    for method_name, (x, y) in data:
        plt.plot(x, y, linewidth=l_width, label=method_name)
    plt.legend()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.show()