import json
import os
import numpy as np
from transform_code.transformations import Transformer

# set up global variable for counting detected radar objects per class
OBJ_CNTS = {}

ANGLE_RES_NEAR_DEG = 3.75
ANGLE_RES_FAR_DEG = 1
ANGLE_RES_NEAR_RAD = ANGLE_RES_NEAR_DEG * np.pi / 180.0
ANGLE_RES_FAR_RAD = ANGLE_RES_FAR_DEG * np.pi / 180.0


def r_phi_array_unscaled(near=False):
    """

    :param near: if peak list near
    :return: (rg_cnt, 17, 2) numpy array for range r and angle phi
    """
    rg_cnt = 90 if near else 200
    az_cnt = 17

    r_phi = np.zeros((rg_cnt, az_cnt, 2))

    angle_res = ANGLE_RES_NEAR_RAD if near else ANGLE_RES_FAR_RAD

    for r_idx in range(rg_cnt):
        for phi_idx in range(az_cnt):
            # polar coordinates
            r = r_idx + 1  # either in [m], [m/2] or [m/4]
            phi = angle_res * (8 - phi_idx)  # in radians

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


def preprocess_output(output):
    if output.shape[0] < 1500:
        # TODO upsample
        pass

    # TODO downsample based on amplitude in output[:, 3]
    # Algo: sort by amplitude
    # select first 1500
    output = output[:1500, :]
    return output


def write_file_for_PointNet(r_phiArray_subarray, amp_subarray, dop_subarray, identity, saveinDirectory):
    # note that all arrays have same width and length
    output = get_output_for_file(r_phiArray_subarray, amp_subarray, dop_subarray, str_out=True) # TODO: wieder auf False
    #output = preprocess_output(output)

    OBJ_CNTS[identity] = 1 + OBJ_CNTS.get(identity, 0)

    out_path = os.path.join('/media/moellya/yannick/data/data_zcomponent', saveinDirectory + identity,
                            '{}_'.format(identity) + '{0:04}.txt'.format(OBJ_CNTS[identity]))

    # pseudo code start
   # points = []
  #  for point in output:  # Achtung geht so evtl nicht bei np arrays
 #       points.append(', '.join([str(e) for e in point]) + '\n')
    # pseudo code end

    with open(out_path, 'w') as f:
        #f.writelines(points)
        f.writelines(output)


def get_output_for_file(r_phiArray_subarray, amp_subarray, dop_subarray, str_out=True):
    output = []

    for k in range(r_phiArray_subarray.shape[1]):
        for j in range(r_phiArray_subarray.shape[2]):
            # TODO r_phi -> x, y
            point = [*r_phiArray_subarray[:, k, j], 0.0, *amp_subarray[:, k, j], *dop_subarray[:, k, j]]
            if str_out:
                output.append(', '.join([str(e) for e in point]) + '\n')
            else:
                output.append(point)

    if not str_out:
        output = np.array(output)
    return output


def draw_array_as_image(array):
    # cv2.imwrite('color_subarray.jpg', amp_subarray)
    # cv2.imshow("amp_subarray", amp_subarray)
    # cv2.waitKey()
    pass


def merge_overlapping_intervals(list):
    if not list:
        return []

    list.sort(key=lambda interval: interval[0])
    merged = [list[0]]
    for current in list:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)

    return merged


def invert_interval_bounderies(obj_list, base_interval):
    previous_right_border = base_interval[0]
    inverted_interval = []

    obj_list.append([base_interval[1], base_interval[1]])

    if not obj_list:
        return base_interval

    for obj in obj_list:
        new_border_right = obj[0]
        new_border_left = previous_right_border
        previous_right_border = obj[1]
        inverted_interval.append([new_border_left, new_border_right])

    return inverted_interval


def cartesian2polar(x, y):
    r = np.sqrt(x * x + y * y)
    phi = np.arctan(y / x)
    return r, phi


def polar2cartesian(phi):
    return np.cos(phi), np.sin(phi)


def look_in_far_radar(phileft_radar, phiright_radar, radar_data, identity):
    r_phiArray = R_PHI_ARRAY_FAR  # create array for range r and angle phi
    phileft_radar += 0.5 * ANGLE_RES_FAR_RAD
    phiright_radar -= 0.5 * ANGLE_RES_FAR_RAD
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
    phileft_radar += 0.5 * 3.75 * ANGLE_RES_NEAR_RAD
    phiright_radar -= 0.5 * 3.75 * ANGLE_RES_NEAR_RAD
    bound_left, bound_right = get_bounderies_for_angle(r_phiArray, phileft_radar, phiright_radar)

    # if bound_right != 0:
    amp = np.array(radar_data['peak_list_near']['amplitude'])
    amp_subarray = create_subarray(amp, bound_left, bound_right)

    dop = np.array(radar_data['peak_list_near']['ego_motion_compensated_doppler'])
    dop_subarray = create_subarray(dop, bound_left, bound_right)

    # create subarray for r and phi and swap axis so that dimensions are congruent to other subarrays (except third one)
    r_phiArray = np.moveaxis(r_phiArray, [2], [0])
    r_phiArray_subarray = create_subarray(r_phiArray, bound_left, bound_right)

    #write_file_for_PointNet(r_phiArray_subarray, amp_subarray, dop_subarray, identity, 'near_data/')


def check_for_enough_resolution(phileft_radar, phiright_radar, near=True):
    ang_res = ANGLE_RES_NEAR_RAD if near else ANGLE_RES_FAR_RAD

    # calculate distance value between phileft and phiright
    distance = phileft_radar - phiright_radar

    return distance > ang_res


def create_no_objects(no_obj_list, radar_data, near=True):
    for phiright_radar, phileft_radar in no_obj_list:
        enough_resolotion = check_for_enough_resolution(phileft_radar, phiright_radar, near)

        if enough_resolotion:
            if near:
                look_in_near_radar(phileft_radar, phiright_radar, radar_data, 'no_obj')
            else:
                look_in_far_radar(phileft_radar, phiright_radar, radar_data, 'no_obj')


def main():
    filenames_radar_files = get_all_radarfiles_from_directory()
    trafo = Transformer()
    print('Creating files for each labelled object from both near and far radar system.')

    far_boundary = R_PHI_ARRAY_FAR[0, 0, 1]
    near_boundary = R_PHI_ARRAY_NEAR[0, 0, 1]
    for counter, current_filename in enumerate(filenames_radar_files, start=1):
        if counter % 10 == 0:
            print('Already ' + str(counter) + ' files used.')

        radar_path = get_radar_path(current_filename)
        with open(radar_path, 'r') as f:
            radar_data = json.load(f)

        label_path = get_label_path(current_filename)
        with open(label_path, 'r') as f:
            label_data = json.load(f)['children'][0]['children'][0]['children']

        obj_list_far = []
        obj_list_near = []

        for label_object in label_data:
            xmin, xmax, y, identity = get_label_cornerpixels(label_object)

            xleft_radar = trafo.img_pixel_to_radar(xmin, y)  # returns 3-tuple
            xright_radar = trafo.img_pixel_to_radar(xmax, y)

            _, phileft_radar = cartesian2polar(xleft_radar[0], xleft_radar[1])
            _, phiright_radar = cartesian2polar(xright_radar[0], xright_radar[1])

            # far radar system
            if (phiright_radar <= far_boundary and phiright_radar >= -far_boundary) or (
                    phileft_radar <= far_boundary and phileft_radar >= -far_boundary):
                look_in_far_radar(phileft_radar, phiright_radar, radar_data, identity)
                obj_list_far.append([phiright_radar, phileft_radar])

            # near radar system
            if (phiright_radar <= near_boundary and phiright_radar >= -near_boundary) or (
                    phileft_radar <= near_boundary and phileft_radar >= -near_boundary):
                look_in_near_radar(phileft_radar, phiright_radar, radar_data, identity)
                obj_list_near.append([phiright_radar, phileft_radar])

        # get obj_lists disjoint -> sort it first, then use disjoint-function
        obj_list_far = merge_overlapping_intervals(obj_list_far)
        obj_list_near = merge_overlapping_intervals(obj_list_near)

        no_obj_list_far = invert_interval_bounderies(obj_list_far,
                                                     [8.5 * (-1) * ANGLE_RES_FAR_RAD, 8.5 * ANGLE_RES_FAR_RAD])
        no_obj_list_near = invert_interval_bounderies(obj_list_near,
                                                      [8.5 * (-1) * ANGLE_RES_NEAR_RAD, 8.5 * ANGLE_RES_NEAR_RAD])
        # zero = invert_interval_bounderies([], [8.5*(-1)*angle_res_near_rad, 8.5*angle_res_near_rad]) # test case

        create_no_objects(no_obj_list_far, radar_data, False)
        create_no_objects(no_obj_list_near, radar_data)

    print(OBJ_CNTS)


if __name__ == '__main__':
    main()
