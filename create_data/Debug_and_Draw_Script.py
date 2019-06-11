import json
import os
import numpy as np
import matplotlib.pyplot as plt
from transform_code.transformations import Transformer
from vru_notvru import *
from draw_2dpointcloud import create_data_array

current_filename= 'radar_-_snow_subsampled:2018-02-12_12-05-18:cam_stereo:left:image_rect:00250_1518433545943946343.json'
radar_path = ''
counter = 0

VRU = ['pedestrian', 'cyclist', 'person_sitting']

ANGLE_RES_NEAR_DEG = 3.75
ANGLE_RES_FAR_DEG = 1
ANGLE_RES_NEAR_RAD = ANGLE_RES_NEAR_DEG * np.pi / 180.0
ANGLE_RES_FAR_RAD = ANGLE_RES_FAR_DEG * np.pi / 180.0

LABEL_MAPPING = {
    identity: 'vru' for identity in VRU
}


def main():
    trafo = Transformer()

    far_boundary = R_PHI_ARRAY_FAR[0, 0, 1]
    near_boundary = R_PHI_ARRAY_NEAR[0, 0, 1]

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

        if identity is not 'vru':
            continue

        else:
            xleft_radar = trafo.img_pixel_to_radar(xmin, y)  # returns 3-tuple
            xright_radar = trafo.img_pixel_to_radar(xmax, y)

            _, phileft_radar = cartesian2polar(xleft_radar[0], xleft_radar[1])
            _, phiright_radar = cartesian2polar(xright_radar[0], xright_radar[1])

            # far radar system
            if (phiright_radar <= far_boundary and phiright_radar >= -far_boundary) or (
                    phileft_radar <= far_boundary and phileft_radar >= -far_boundary):
                look_in_far_radar(phileft_radar, phiright_radar, radar_data, identity, counter)
                obj_list_far.append([phiright_radar, phileft_radar])

            # near radar system
            if (phiright_radar <= near_boundary and phiright_radar >= -near_boundary) or (
                    phileft_radar <= near_boundary and phileft_radar >= -near_boundary):
                look_in_near_radar(phileft_radar, phiright_radar, radar_data, identity, counter)
                obj_list_near.append([phiright_radar, phileft_radar])

    # get obj_lists disjoint -> sort it first, then use disjoint-function
    obj_list_far = merge_overlapping_intervals(obj_list_far)
    obj_list_near = merge_overlapping_intervals(obj_list_near)

    no_obj_list_far = invert_interval_bounderies(obj_list_far,
                                                 [8.5 * (-1) * ANGLE_RES_FAR_RAD, 8.5 * ANGLE_RES_FAR_RAD])
    no_obj_list_near = invert_interval_bounderies(obj_list_near,
                                                  [8.5 * (-1) * ANGLE_RES_NEAR_RAD, 8.5 * ANGLE_RES_NEAR_RAD])
    # zero = invert_interval_bounderies([], [8.5*(-1)*angle_res_near_rad, 8.5*angle_res_near_rad]) # test case

    create_no_objects(no_obj_list_far, radar_data, counter, False)
    create_no_objects(no_obj_list_near, radar_data, counter)



    #################### DRAW FILE #####################



    xy_amp = create_data_array()

    plt.title("VRU Detection in Radar Data")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    marker_size = 5

    #for point in xyz:
    #    c = [point[2], point[3], point[4]]
    #    plt.scatter(point[1] * (-1), point[0], marker_size, c)

    # c = (xyz[:, 2], xyz[:, 3], xyz[:, 4])

    plt.scatter(xy_amp[:, 1] * (-1), xy_amp[:, 0], marker_size, c=xy_amp[:, 2])  # *(-1) as one has to swap sign of phileft and phiright
    cbar = plt.colorbar()
    cbar.set_label("Amplitude [16bit number]", labelpad=+1)
    plt.show()


if __name__ == '__main__':
    main()


