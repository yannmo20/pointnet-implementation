import json
import os
import numpy as np
from transform_code.transformations import Transformer

# set up global variable for counting detected radar objects per class
OBJ_CNTS = {}
COUNTER = 0


def r_phi_array_unscaled(near=False):
    """

    :param near: if peak list near
    :return: (rg_cnt, 17, 2) numpy array for range r and angle phi
    """
    rg_cnt = 90 if near else 200
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


R_PHI_ARRAY_FAR = r_phi_array_unscaled(near=False)
R_PHI_ARRAY_NEAR = r_phi_array_unscaled(near=True)
print()


def get_all_radarfiles_from_directory():
    return os.listdir('/media/moellya/yannick/raw_data/')


def get_radar_path(current_filename):
    radar_path = '/media/moellya/yannick/raw_data/' + current_filename
    return radar_path


def get_label_path(current_filename):
    current_filename = current_filename[8:]  # cut off prefix 'radar_-_'of label file
    label_path = '/media/moellya/yannick/labels/' + current_filename
    return label_path


def get_bounderies_for_angle(r_phiArray, phileft_radar, phiright_radar):
    bound_left = 0  # init for error handling
    for k in range(r_phiArray.shape[1]):
        if phileft_radar >= r_phiArray[0][k][1]:
            if k == 0:
                bound_left = 0
            else:
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
    output = get_output_for_file(r_phiArray_subarray, amp_subarray, dop_subarray)

    OBJ_CNTS[identity] = 1 + OBJ_CNTS.get(identity, 0)

    out_path = os.path.join('data_zcomponent', saveinDirectory + identity,
                            '{}_'.format(identity) + '{0:04}.txt'.format(OBJ_CNTS[identity]))

    with open(out_path, 'w') as f:
        f.writelines(output)

    # print number of created files
    if COUNTER % 100 == 0:
        print('Created ' + str(COUNTER) + ' files.')

    COUNTER += 1


def get_output_for_file(r_phiArray_subarray, amp_subarray, dop_subarray):
    output = []

    for k in range(r_phiArray_subarray.shape[1]):
        for j in range(r_phiArray_subarray.shape[2]):
            # TODO r_phi -> x, y
            output.append(str(r_phiArray_subarray[0][k][j]) + ', ' + str(r_phiArray_subarray[1][k][j]) + ', ' + str(
                0.0) + ',' +
                          str(amp_subarray[0][k][j]) + ', ' + str(amp_subarray[1][k][j]) + ', ' + str(
                amp_subarray[2][k][j]) + ', '
                          + str(dop_subarray[0][k][j]) + ', ' + str(dop_subarray[1][k][j]) + ', ' + str(
                dop_subarray[2][k][j]) + '\n')

    return output


def draw_array_as_image(array):
    # cv2.imwrite('color_subarray.jpg', amp_subarray)
    # cv2.imshow("amp_subarray", amp_subarray)
    # cv2.waitKey()
    pass


def merge_overlapping_intervals(list):
    list.sort(key=lambda interval: interval[0])
    merged = [list[0]]
    for current in list:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)

    return merged


def cartesian2polar(x, y):
    r = np.sqrt(x * x + y * y)
    phi = np.arctan(y / x)
    return r, phi


def look_in_far_radar(phileft_radar, phiright_radar, radar_data, identity):
    r_phiArray = R_PHI_ARRAY_FAR  # create array for range r and angle phi
    phileft_radar += 0.5 * 0.01745329
    phiright_radar -= 0.5 * 0.01745329
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


def look_in_near_radar(phileft_radar, phiright_radar, radar_data, identity):
    r_phiArray = R_PHI_ARRAY_NEAR  # create array for range r and angle phi
    phileft_radar += 0.5 * 3.75 * 0.01745329
    phiright_radar -= 0.5 * 3.75 * 0.01745329
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


def get_no_obj(obj_list, radar_data, near=True):
    angle_res = 3.75 if near else 1
    no_obj_list = []

    if len(obj_list) > 1:
        for k in range(len(obj_list) - 1):
            if obj_list[k + 1][0] - obj_list[k][1] >= 0.01745329 * angle_res:  # size of far angle resolution
                no_obj_list.append([obj_list[k][1], obj_list[k + 1][0]])

        if obj_list[0][0] > -0.12217305 * angle_res:  # second boundery from the right side -> right side of radar cone
            no_obj_list.append([-0.13962634 * angle_res, obj_list[0][0]])

        if obj_list[len(obj_list) - 1][1] < 0.12217305 * angle_res:  # left side of radar cone
            no_obj_list.append([obj_list[len(obj_list) - 1][1], 0.13962634 * angle_res])

        for k in range(len(no_obj_list)):
            if near:
                look_in_near_radar(no_obj_list[k][1], no_obj_list[k][0], radar_data, 'no_obj')
            else:
                look_in_far_radar(no_obj_list[k][1], no_obj_list[k][0], radar_data, 'no_obj')

    elif len(obj_list) == 1:
        if near:
            look_in_near_radar(0.13962634 * angle_res, obj_list[0][1], radar_data, 'no_obj')
            look_in_near_radar(obj_list[0][0], -0.13962634 * angle_res, radar_data, 'no_obj')
        else:
            look_in_far_radar(0.13962634 * angle_res, obj_list[0][1], radar_data, 'no_obj')
            look_in_far_radar(obj_list[0][0], -0.13962634 * angle_res, radar_data, 'no_obj')

    else:
        if near:
            look_in_near_radar(0.13962634 * angle_res, -0.13962634 * angle_res, radar_data, 'no_obj')
        else:
            look_in_far_radar(0.13962634 * angle_res, -0.13962634 * angle_res, radar_data, 'no_obj')


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

        obj_list_far = []  # TODO fill
        obj_list_near = []  # TODO fill

        for label_object in label_data:
            xmin, xmax, y, identity = get_label_cornerpixels(label_object)

            xleft_radar = trafo.img_pixel_to_radar(xmin, y)  # returns 3-tuple
            xright_radar = trafo.img_pixel_to_radar(xmax, y)

            _, phileft_radar = cartesian2polar(xleft_radar[0], xleft_radar[1])
            _, phiright_radar = cartesian2polar(xright_radar[0], xright_radar[1])

            # far radar system
            if (phiright_radar <= 0.13962634014954636 and phiright_radar >= -0.13962634014954636) or (
                    phileft_radar <= 0.13962634014954636 and phileft_radar >= -0.13962634014954636):
                look_in_far_radar(phileft_radar, phiright_radar, radar_data, identity)
                obj_list_far.append([phiright_radar, phileft_radar])

            # near radar system
            if (phiright_radar <= 0.5235987755982988 and phiright_radar >= -0.5235987755982988) or (
                    phileft_radar <= 0.5235987755982988 and phileft_radar >= -0.5235987755982988):
                look_in_near_radar(phileft_radar, phiright_radar, radar_data, identity)
                obj_list_near.append([phiright_radar, phileft_radar])

        # get obj_lists disjoint -> sort it first, then use disjoint-function
        if len(obj_list_far) > 1:
            obj_list_far = merge_overlapping_intervals(obj_list_far)
        if len(obj_list_near) > 1:
            obj_list_near = merge_overlapping_intervals(obj_list_near)

        get_no_obj(obj_list_far, radar_data, False)
        get_no_obj(obj_list_near, radar_data)

    print(OBJ_CNTS)


if __name__ == '__main__':
    main()
