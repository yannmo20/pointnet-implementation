import json
import os
import numpy as np
from transform_code.transformations import Transformer

# set up global variables for counting detected radar objects per class
van = 0
car = 0
truck = 0
cyclist = 0
motorcycle = 0
bicycle = 0
person_sitting = 0
tram = 0
bus = 0
pedestrian = 0
noObject = 0


def get_all_radarfiles_from_directory():
    return os.listdir('/media/moellya/yannick/raw_data/')


def get_radar_path(current_filename):
    radar_path = '/media/moellya/yannick/raw_data/' + current_filename
    return radar_path


def get_label_path(current_filename):
    current_filename = current_filename[8:] # cut off prefix 'radar_-_'of label file
    label_path = '/media/moellya/yannick/labels/' + current_filename
    return label_path


def get_bounderies_for_angle(r_phiArray, phileft_radar, phiright_radar):
    for k in range(r_phiArray.shape[1]):
        if phileft_radar >= r_phiArray[0][k][1]:
            bound_left = k - 1
            break

    for k in range(r_phiArray.shape[1]):
        if k == r_phiArray.shape[1] - 1: bound_right = k
        if phiright_radar >= r_phiArray[0][k][1]:
            bound_right = k - 1
            break

    return bound_left, bound_right


def get_label_cornerpixels(label_object):
    xmin = label_object['maxcol']  # x left
    xmax = xmin + label_object['mincol']  # x right
    ymin = label_object['maxrow']  # y top
    ymax = label_object['minrow'] + ymin  # y bottom

    y = 0.5 * (ymax - ymin)  # mid-height if label
    identity = label_object['identity']

    return xmin, xmax, y, identity


def create_subarray(array, bound_left, bound_right):
    return array[:, :, bound_left:bound_right + 1]


def write_file_for_PointNet(r_phiArray_subarray, amp_subarray, dop_subarray, identity, saveinDirectory):
    # note that all arrays have same width and length

    global van, car, truck, cyclist, motorcycle, bicycle, person_sitting, tram, bus, pedestrian, noObject

    dimensions = amp_subarray.shape
    output = get_output_for_file(r_phiArray_subarray, amp_subarray, dop_subarray, dimensions)

    if identity == 'van':
        van += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(van) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'car':
        car += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(car) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'truck':
        truck += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(truck) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'cyclist':
        cyclist += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(cyclist) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'motorcycle':
        motorcycle += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(motorcycle) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'bicycle':
        bicycle += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(bicycle) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'person_sitting':
        person_sitting += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(person_sitting) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'tram':
        tram += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(tram) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'bus':
        bus += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(bus) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'pedestrian':
        pedestrian += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(pedestrian) + '.txt', 'w') as f:
            f.write(output)

    elif identity == 'noObject':
        noObject += 1
        with open('data_dense_car/' + saveinDirectory + identity + '_' + '{0:04}'.format(noObject) + '.txt', 'w') as f:
            f.write(output)


def get_output_for_file(r_phiArray_subarray, amp_subarray, dop_subarray, dimensions):
    k = 0
    j = 0
    output = ''

    while k < dimensions[1]:
        while j < dimensions[2]:
            output += str(r_phiArray_subarray[0][k][j]) + ', ' + str(r_phiArray_subarray[1][k][j]) + ', ' + str(
                amp_subarray[0][k][j]) + ', ' + str(amp_subarray[1][k][j]) + ', ' + str(
                amp_subarray[2][k][j]) + ', ' + str(dop_subarray[0][k][j]) + ', ' + str(
                dop_subarray[1][k][j]) + ', ' + str(dop_subarray[0][k][j]) + '\n'
            j += 1
        j = 0
        k += 1

    return output


def draw_array_as_image(array):
    # cv2.imwrite('color_subarray.jpg', amp_subarray)
    # cv2.imshow("amp_subarray", amp_subarray)
    # cv2.waitKey()
    pass


def cartesian2polar(x, y):
    r = np.sqrt(x * x + y * y)
    phi = np.arctan(y / x)
    return r, phi


def r_phi_array_unscaled(near=False):
    """

    :param near: if peak list near
    :return: (rg_cnt, 17, 2) numpy array for range r and angle phi
    """
    rg_cnt = 100 if near else 200
    az_cnt = 17

    r_phi = np.zeros((rg_cnt, az_cnt, 2))

    angle_res = 3.75 if near else 1

    for r_idx in range(rg_cnt):
        for phi_idx in range(az_cnt):
            # polar coordinates
            r = r_idx + 1  # either in [m], [m/2] or [m/4]
            phi = angle_res * (8 - phi_idx) * np.pi / 180.0  # in radians

            r_phi[r_idx, phi_idx, :] = r, phi

            # when converting to cartesian: x is forward and y is left
            # Top down view of coordinate system (the sensor is at 0):
            #
            #                                 ^ x
            #                                 |
            #                       X         |
            #                        \ <----> |
            #                         \  phi  |
            #                          \      |
            #                        r  \     |
            #                            \    |
            #                             \   |
            #                              \  |
            #                               \ |
            #                                \|
            # <-------------------------------0
            # y
    return r_phi


def look_in_far_radar(near, phileft_radar, phiright_radar, radar_data, identity):
    r_phiArray = r_phi_array_unscaled(near)  # create array for range r and angle phi
    bound_left, bound_right = get_bounderies_for_angle(r_phiArray, phileft_radar, phiright_radar)

    # if bound_right != 0:
    amp = np.array(radar_data['peak_list_far']['amplitude'])
    amp_subarray = create_subarray(amp, bound_left, bound_right)

    dop = np.array(radar_data['peak_list_far']['ego_motion_compensated_doppler'])
    dop_subarray = create_subarray(dop, bound_left, bound_right)

    # create subarray for r and phi and swap axis so that dimensions are congruent to other subarrays (except third one)
    r_phiArray = np.moveaxis(r_phiArray, [2], [0])
    r_phiArray_subarray = create_subarray(r_phiArray, bound_left, bound_right)

    write_file_for_PointNet(r_phiArray_subarray, amp_subarray, dop_subarray, identity, 'far_data/')


def look_in_near_radar(near, phileft_radar, phiright_radar, radar_data, identity):
    r_phiArray = r_phi_array_unscaled(near)  # create array for range r and angle phi
    bound_left, bound_right = get_bounderies_for_angle(r_phiArray, phileft_radar, phiright_radar)

    # if bound_right != 0:
    amp = np.array(radar_data['peak_list_near']['amplitude'])
    amp_subarray = create_subarray(amp, bound_left, bound_right)

    dop = np.array(radar_data['peak_list_near']['ego_motion_compensated_doppler'])
    dop_subarray = create_subarray(dop, bound_left, bound_right)

    # create subarray for r and phi and swap axis so that dimensions are congruent to other subarrays (except third one)
    r_phiArray = np.moveaxis(r_phiArray, [2], [0])
    r_phiArray_subarray = create_subarray(r_phiArray, bound_left, bound_right)

    write_file_for_PointNet(r_phiArray_subarray, amp_subarray, dop_subarray, identity, 'near_data/')


def main():
    filenames_radar_files = get_all_radarfiles_from_directory()
    trafo = Transformer()
    print('Creating files for each labelled object from both near and far radar system.')
    counter = 0

    for current_filename in filenames_radar_files:

        radar_path = get_radar_path(current_filename)
        with open(radar_path, 'r') as f:
            radar_data = json.load(f)

        label_path = get_label_path(current_filename)
        with open(label_path, 'r') as f:
            label_data = json.load(f)['children'][0]['children'][0]['children']


        for label_object in label_data:
            # print number of created files
            if counter % 100 == 0:
                print('Created ' + str(counter) + ' files.')

            counter += 1
            xmin, xmax, y, identity = get_label_cornerpixels(label_object)

            xleft_radar = trafo.img_pixel_to_radar(xmin, y)  # returns 3-tuple
            xright_radar = trafo.img_pixel_to_radar(xmax, y)

            r1, phileft_radar = cartesian2polar(xleft_radar[0], xleft_radar[1])
            r2, phiright_radar = cartesian2polar(xright_radar[0], xright_radar[1])

            r = r1 if (r1 > r2) else r2  # take greater value of r1 and r2 for range r
            r = int(np.ceil(r))

            # use far radar system
            if (phiright_radar <= 0.13962634014954636 and phiright_radar >= -0.13962634014954636) or (phileft_radar <= 0.13962634014954636 and phileft_radar >= -0.13962634014954636):
                near = False
                look_in_far_radar(near, phileft_radar, phiright_radar, radar_data, identity)

            # else: use near radar system -> no doubling for same object in both radar systems
            elif (phiright_radar <= 0.5235987755982988 and phiright_radar >= -0.5235987755982988) or (phileft_radar <= 0.5235987755982988 and phileft_radar >= -0.5235987755982988):
                near = True
                look_in_near_radar(near, phileft_radar, phiright_radar, radar_data, identity)


if __name__ == '__main__':
    main()
