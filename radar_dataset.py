'''
    Radar dataset to be used for training (data are generated radar subcones from recorded traffic scenarios.
'''

import os
import os.path
import numpy as np
import sys
# import provider

counter = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


def pc_normalize_far(xyz):
    xyz[:, 0] /= 200.0
    xyz[:, 1] /= 29.561882225922126
    xyz[:, 2] /= 200.0
    return xyz


def pc_normalize_near(xyz):
    xyz[:, 0] /= 90.0
    xyz[:, 1] /= 0.0  # todo correct value
    xyz[:, 2] /= 200.0
    return xyz


def normalize_features(features):
    for k in range(3):
        features[:, k] /= 65535.0  # equals float(np.power(2, 16)-1)
        features[:, k + 3] /= 200.0
    return features


class RadarDataset():
    def __init__(self, root, batch_size=32, npoints=1024, split='train', normalize=True,
                 modelnet10=False, cache_size=100000, shuffle=None):  # TODO nr of points
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints  # TODO
        self.normalize = normalize  # TODO
        self.catfile = '/media/moellya/yannick/data/data_vru_notvru/far_data/shape_names.txt'
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(
            '/media/moellya/yannick/data/data_vru_notvru/far_data/radar_train.txt')]
        shape_ids['test'] = [line.rstrip() for line in open(
            '/media/moellya/yannick/data/data_vru_notvru/far_data/radar_val.txt')]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i][5:], os.path.join(self.root, shape_names[i][5:], shape_ids[split][i]) + '.txt')
                         for i in range(len(shape_ids[split]))]  # [5:0] cuts off scenario_nr + hyphen

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        if shuffle is None:
            if split == 'train':
                self.shuffle = True
            else:
                self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _augment_batch_data(self, batch_data):
        return batch_data
        # if self.normal_channel:
        #     rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
        #     rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        # else:
        #     rotated_data = provider.rotate_point_cloud(batch_data)
        #     rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        #
        # jittered_data = provider.random_scale_point_cloud(batch_data[:, :, :3])
        # jittered_data = provider.shift_point_cloud(jittered_data)
        # jittered_data = provider.jitter_point_cloud(jittered_data)
        # batch_data[:, :, :3] = jittered_data
        # return provider.shuffle_points(batch_data)

    def _get_item(self, index):
        global counter
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            # Take the first npoints
            # print(point_set.shape, cls, fn)

            # counter += 1
            # if counter % 100 == 0:
            #     print('Already ' + str(counter) + ' objects read and stored into cache.')

            point_set = point_set[0:self.npoints, :]

            # normalize
            point_set[:, 0:3] = pc_normalize_far(point_set[:, 0:3])
            point_set[:, 3:] = normalize_features(point_set[:, 3:])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        return 9

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath) + self.batch_size - 1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        # returned dimension may be smaller than self.batch_size!
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.npoints, self.num_channel()))
        batch_label = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps, cls = self._get_item(self.idxs[i + start_idx])
            batch_data[i] = ps
            batch_label[i] = cls
        self.batch_idx += 1
        if augment: batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label


if __name__ == '__main__':
    d = RadarDataset(root='../data/radar_dataset', split='test')
    print(d.shuffle)
    print(len(d))
    import time

    tic = time.time()
    for i in range(10):
        ps, cls = d[i]
    print(time.time() - tic)
    print(ps.shape, type(ps), cls)

    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
