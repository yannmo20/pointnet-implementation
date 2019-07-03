import json

import numpy as np

from transform_code.cameramodels import PinholeCameraModel


class Transformer:
    def __init__(self):
        with open('transform_code/calib_cam_stereo_left.json', 'r') as f:
            calib = json.load(f)
        self.__cam = PinholeCameraModel()
        self.__cam.fromCameraInfo(calib)

        # self.__mat44 = np.array([[1.86326606e-04, -6.85600857e-03, 9.99976480e-01, -1.83312000e+00],
        #                          [9.99998976e-01, 1.42024365e-03, -1.76593366e-04, 8.69610000e-02],
        #                          [-1.41899952e-03, 9.99975489e-01, 6.85626617e-03, 6.10750000e-01],
        #                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.mat44 = np.array([[1.86326606e-04, 6.85600857e-03, 9.99976480e-01, -1.83312000e+00],
                               [-9.99998976e-01, 1.42024365e-03, 1.76593366e-04, 8.69610000e-02],
                               [-1.41899952e-03, -9.99975489e-01, 6.85626617e-03, 6.10750000e-01],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        self.cam2radar = np.zeros_like(self.mat44)
        self.cam2radar[:3, :3] = self.mat44[:3, :3].transpose()
        self.cam2radar[:3, 3] = -self.mat44[:3, 3]
        self.cam2radar[3, 3] = 1.0

    def img_pixel_to_radar(self, x, y):
        uv = (x, y)
        x, y, z = self.__cam.projectPixelTo3dRay(uv)
        scale = 100
        x *= scale
        y *= scale
        z *= scale

        return self.__cam_to_radar(x, y, z)

    def __cam_to_radar(self, x, y, z):
        xyz = tuple(np.dot(self.mat44, np.array([x, y, z, 1.0])))[:3]
        return xyz

    def radar_to_cam(self, x, y, z):
        xyz = tuple(np.dot(self.cam2radar, np.array([z, -x, -y, 1.0])))[:3]
        return xyz

    def radar_to_pixel(self, x, y, z):
        return self.__cam.project3dToPixel(self.radar_to_cam(x, y, z))


def main():
    trafo = Transformer()
    # print(trafo.img_pixel_to_radar(512, 960))


if __name__ == '__main__':
    main()
    trafo = Transformer()
    print(trafo.radar_to_cam(0, 0, 0))
    print(trafo.radar_to_cam(1000, 0, 0))
    print(trafo.radar_to_cam(0, 1, 0))
