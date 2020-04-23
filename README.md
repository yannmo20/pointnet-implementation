# pointnet-implementation
PointNet architecture and data preprocessing for basic object classification:
 
The idea is to use recorded radar data from traffic scenarios and perform classifications 
for vulnerable road users (VRUs) like cyclists and pedestrians.

The main part in preprocessing deals with cutting out the matching radar subcones of the VRU
objects, which were already labelled in camera pictures (the measurement vehicle captured
the radar scenery and the camera picture nearby at the same time).

This data is then used to perform classifications into VRU and non-VRU by using radar 
amplitudes and Doppler velocities as characteristic VRU features.

(mainly from April to August, 2019)
