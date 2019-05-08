import os

sensor_type = 'far_data'
path = '/home/moellya/PycharmProjects/pointnet-implementation/data/' + sensor_type

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            addstring = os.path.join(r, file)

            if sensor_type == 'near_data':
                substring_addstring = addstring[69:]  # cut off prefix till start of subdirecectory; 69 for near_data;
            elif sensor_type == 'far_data':
                substring_addstring = addstring[68:]

            if substring_addstring != 'shape_names.txt' and substring_addstring != 'filelist.txt':
                if sensor_type == 'near_data':
                    files.append(substring_addstring)  # cut off prefix till start of subdirecectory; 69 for near_data;
                elif sensor_type == 'far_data':
                    files.append(substring_addstring)

f = open(sensor_type + '/filelist.txt', 'w')
f.close()

print('Writing names into file, please wait...')
for f in files:
    # print(f)
    with open(sensor_type + '/filelist.txt', 'a') as filelist:
        filelist.write(f + '\n')

print('Writing finished.')
