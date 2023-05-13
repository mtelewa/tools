import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()


def animate(i):
    x_vals.append(next(index))
    y_vals.append(random.randint(0, 5))
    plt.cla()

    plt.plot(x_vals, y_vals)

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
# plt.show()

# saving to mp4 using ffmpeg writer
writervideo = FFMpegWriter(fps=60)
ani.save('increasingStraightLine.mp4', writer=writervideo)
plt.close()
