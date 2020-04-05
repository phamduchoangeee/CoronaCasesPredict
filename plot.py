import matplotlib.pyplot as plt


def plot_data(date, Y, output):
    plt.plot(date, Y, 'ro', date, output, 'b-')

    plt.ylabel('Infected cases in Japan')
    plt.xlabel('Date')
    plt.show()