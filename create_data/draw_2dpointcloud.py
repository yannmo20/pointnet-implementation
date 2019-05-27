import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import matplotlib.lines as mlines
import numpy as np


def create_data_array():
    datapoints = get_radarfile('/lhome/moellya/Desktop/test.txt')  # TODO CHNGE!
    #datapoints = get_radarfile('/media/moellya/yannick/data/data_vru_notvru_ziszero/far_data/not_vru/3162-not_vru_9380.txt')  # TODO CHNGE!
    check_for_multiple_drawing = set()
    xy_amp = np.zeros([len(datapoints), 5])

    for pointnr, point in enumerate(datapoints):

        point_objects = point.split(',')
        point_objects[len(point_objects) - 1] = point_objects[len(point_objects) - 1][:-2]  # cut off '\n'

        if (point_objects[0], point_objects[1]) not in check_for_multiple_drawing:
            check_for_multiple_drawing.add((point_objects[0], point_objects[1]))

            for k in range(2):
                xy_amp[pointnr][k] = point_objects[k]

            # three amplitudes, +1 for full 16bit -> then to 8bit and -1 back -> go to RGB representation
            for k in range(3):
                xy_amp[pointnr][2 + k] = float(point_objects[3 + k]) / (np.power(2, 16) - 1)

        else:
            continue

    xy_amp = xy_amp[~np.all(xy_amp == 0, axis=1)]
    return xy_amp


def get_radarfile(path):
    with open(path, 'r') as file:
        return file.readlines()


def main():
    font = {'family': 'Computer Modern',
            'size': 12}
    rcParams['font.family'] = 'Computer Modern'

    plt.rc('font', **font)
    plt.rc('axes', axisbelow=True)
    xy_amp = create_data_array()
    #plt.scatter(xyz[:, 1], xyz[:, 0])

    fpath = '/lhome/moellya/Downloads/ComputerModern/cmunrm.ttf'
    prop = fm.FontProperties(fname=fpath)
    fname = os.path.split(fpath)[1]

    plt.title("Detection in Radar Data".format(fname), fontproperties=prop)
    plt.xlabel("x [m]".format(fname), fontproperties=prop)
    plt.ylabel("y [m]".format(fname), fontproperties=prop)
    plt.grid(color='whitesmoke')

    marker_size = 3
    plt.scatter(xy_amp[:, 1] * (-1), xy_amp[:, 0], marker_size, c=xy_amp[:, 2])

    cbar = plt.colorbar()
    cbar.set_label("Amplitude [16bit number]".format(fname), fontproperties=prop, labelpad=+1)

    plt.show()


if __name__ == '__main__':
    main()
