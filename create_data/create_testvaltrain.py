import os
import numpy as np

sensor_type = 'far_data'
path = '/media/moellya/yannick/data/data_zcomponent/' + sensor_type


def clear_textfiles():
    with open(path + '/radar_train.txt', 'w') as f:
        pass
    with open(path + '/radar_test.txt', 'w') as f:
        pass


def write_into_train_file(output):
    with open(path + '/radar_train.txt', 'a+') as f:
        f.writelines(output + '\n')


def write_into_test_file(output):
    with open(path + '/radar_test.txt', 'a+') as f:
        f.writelines(output + '\n')


def go_through_filelist(OBJ_CNTS):
    f = open(path + '/filelist.txt', 'r')
    for line in f.readlines():
        line = str(line)
        if line[0:4] == 'tram':
            OBJ_CNTS['tram'] = 1 + OBJ_CNTS.get('tram', 0)
        elif line[0:7] == ('bicycle'):
            OBJ_CNTS['bicycle'] = 1 + OBJ_CNTS.get('bicycle', 0)
        elif line[0:3] == ('bus'):
            OBJ_CNTS['bus'] = 1 + OBJ_CNTS.get('bus', 0)
        elif line[0:3] == ('car'):
            OBJ_CNTS['car'] = 1 + OBJ_CNTS.get('car', 0)
        elif line[0:7] == ('cyclist'):
            OBJ_CNTS['cyclist'] = 1 + OBJ_CNTS.get('cyclist', 0)
        elif line[0:10] == ('motorcycle'):
            OBJ_CNTS['motorcycle'] = 1 + OBJ_CNTS.get('motorcycle', 0)
        elif line[0:6] == ('no_obj'):
            OBJ_CNTS['no_obj'] = 1 + OBJ_CNTS.get('no_obj', 0)
        elif line[0:10] == ('pedestrian'):
            OBJ_CNTS['pedestrian'] = 1 + OBJ_CNTS.get('pedestrian', 0)
        elif line[0:14] == ('person_sitting'):
            OBJ_CNTS['person_sitting'] = 1 + OBJ_CNTS.get('person_sitting', 0)
        elif line[0:5] == ('truck'):
            OBJ_CNTS['truck'] = 1 + OBJ_CNTS.get('truck', 0)
        elif line[0:3] == ('van'):
            OBJ_CNTS['van'] = 1 + OBJ_CNTS.get('van', 0)
    f.close()

    return OBJ_CNTS


def get_output(slicing_nr, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line):
    OBJ_CNTS[identity] = 1 + OBJ_CNTS.get(identity, 0)
    if OBJ_CNTS[identity] <= OBJ_TRAIN[identity]:
        write_into_test_file(line[slicing_nr:])
    else:
        write_into_train_file(line[slicing_nr:])


def main():
    # split into train (70%) and test(30%)
    train_part = 0.7

    clear_textfiles()

    OBJ_CNTS = {}
    OBJ_TRAIN = {}
    OBJ_TEST = {}

    OBJ_CNTS = go_through_filelist(OBJ_CNTS)
    OBJ_TRAIN = dict(OBJ_CNTS)
    OBJ_TEST = dict(OBJ_CNTS)

    for key in OBJ_TRAIN:
        OBJ_TRAIN[key] = int(np.round(OBJ_TRAIN[key]) * train_part)

    for key in OBJ_TEST:
        OBJ_TEST[key] = OBJ_CNTS[key] - OBJ_TRAIN[key]

    print(OBJ_TRAIN)
    print(OBJ_TEST)
    OBJ_CNTS = {}

    f = open(path + '/filelist.txt', 'r')
    for line in f.readlines():
        line = line[:len(line) - 5]  # cut off '.txt'

        if line[0:4] == 'tram':
            identity = 'tram'
            get_output(5, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:7] == ('bicycle'):
            identity = 'bicycle'
            get_output(8, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:3] == ('bus'):
            identity = 'bus'
            get_output(4, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:3] == ('car'):
            identity = 'car'
            get_output(4, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:7] == ('cyclist'):
            identity = 'cyclist'
            get_output(8, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:10] == ('motorcycle'):
            identity = 'motorcycle'
            get_output(11, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:6] == ('no_obj'):
            identity = 'no_obj'
            get_output(7, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:10] == ('pedestrian'):
            identity = 'pedestrian'
            get_output(11, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:14] == ('person_sitting'):
            identity = 'person_sitting'
            get_output(15, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:5] == ('truck'):
            identity = 'truck'
            get_output(6, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

        elif line[0:3] == ('van'):
            identity = 'van'
            get_output(4, identity, OBJ_CNTS, OBJ_TRAIN, OBJ_TEST, line)

    f.close()


if __name__ == '__main__':
    main()
