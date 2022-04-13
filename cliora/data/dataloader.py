import os
import h5py
import torch
from torch.utils.data import Sampler

import numpy as np
import pickle as pkl
import json
from cliora.logging.configuration import get_logger

class FixedLengthBatchSampler(Sampler):

    def __init__(self, data_source, batch_size, include_partial=False, rng=None, maxlen=None,
                 length_to_size=None):
        self.data_source = data_source
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial
        self.length_to_size = length_to_size
        self._batch_size_cache = { 0: self.batch_size }
        self.logger = get_logger()

    def get_batch_size(self, length):
        if self.length_to_size is None:
            return self.batch_size
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]
        start = max(self._batch_size_cache.keys())
        batch_size = self._batch_size_cache[start]
        for n in range(start+1, length+1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]
            self._batch_size_cache[n] = batch_size
        return batch_size

    def reset(self):
        """
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """

        # Record the lengths of each example.
        length_map = {}
        for i in range(len(self.data_source)):
            x = self.data_source.dataset[i]
            length = len(x)

            if self.maxlen is not None and self.maxlen > 0 and length > self.maxlen:
                continue

            length_map.setdefault(length, []).append(i)

        # Shuffle the order.
        for length in length_map.keys():
            self.rng.shuffle(length_map[length])

        # Initialize state.
        state = {}
        for length, arr in length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size
            surplus = nbatches * batch_size < len(arr)
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = []
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus']:
                    order += [length]

        self.rng.shuffle(order)

        self.length_map = length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1

        length = self.order[index]
        batch_size = self.get_batch_size(length)
        position = self.state[length]['position'] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position
        self.index = index

        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        return len(self.order)


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return index, item, np.zeros(1), np.zeros(1), np.zeros(1)

    def __len__(self):
        return len(self.dataset)


class COCODataset(torch.utils.data.Dataset):

    def __init__(self, dataset, img_ids=None):
        self.dataset = dataset
        self.img_ids = img_ids
        self.data_path = './coco_data'

    def __getitem__(self, index):
        item = self.dataset[index]
        # img_id = self.img_ids[index]
        # obj_data = np.load(os.path.join(self.data_path, 'det_feats/{}.npy'.format(img_id)), allow_pickle=False)
        # obj_feats = obj_data[:, :-4]
        # boxes = obj_data[:, -4:]

        obj_feats = np.zeros(1).astype(np.int32) - 1
        boxes = np.zeros(1).astype(np.int32) - 1
        obj_cates = np.zeros(1).astype(np.int32) - 1
        return index, item, obj_feats, boxes, obj_cates

    def __len__(self):
        return len(self.dataset)


# class FlickrDataset(torch.utils.data.Dataset):
#
#     def __init__(self, dataset, img_ids=None, mode='train'):
#         self.dataset = dataset
#         self.img_ids = img_ids
#         self.mode = mode
#         self.data_path = './flickr_data/'
#
#     def __getitem__(self, index):
#         item = self.dataset[index]
#         img_id = self.img_ids[index]
#
#         obj_feats = np.zeros([36, 2048]).astype(np.float32)
#         boxes = np.zeros([36, 4]).astype(np.float32) - 1
#         obj_cates = np.zeros([36]).astype(np.int32) - 1
#
#         if self.pkl:
#             obj_data = self.det_res[img_id]
#             num_box = min(len(obj_data['bbox']), 36)
#             obj_feats[:num_box] = obj_data['feats'][:num_box]
#             boxes[:num_box] = obj_data['bbox'][:num_box]
#             obj_cates[:num_box] = obj_data['class'][:num_box]
#         else:
#             obj_data = np.load(self.data_path+'flickr_feat/{}.npy'.format(img_id), allow_pickle=False)
#             num_box = min(obj_data.shape[0], 36)
#
#             obj_feats[:num_box] = obj_data[:num_box, :2048].astype(np.float32)
#             boxes[:num_box] = obj_data[:num_box, 2048:-1].astype(np.float32)
#             obj_cates[:num_box] = obj_data[:num_box, -1].astype(np.int32)
#
#         return index, item, obj_feats, boxes, obj_cates
#
#     def __len__(self):
#         return len(self.dataset)


class FlickrDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, img_ids=None, mode='train'):
        self.dataset = dataset
        self.img_ids = img_ids
        self.mode = mode
        self.data_path = './flickr_data/flickr_feat_maf/'
        self.imgid2idx = pkl.load(open(self.data_path+f"{mode}_imgid2idx.pkl", "rb"))
        self.detection_dict = json.load(open(self.data_path+f"{mode}_detection_dict.json"))
        obj_vocab = open(self.data_path+"objects_vocab.txt").readlines()
        self.obj2ind = {obj.strip():idx for idx,obj in enumerate(obj_vocab)}
        with h5py.File(self.data_path+f"{mode}_features_compress.hdf5", "r") as hdf5_file:
            self.features = np.array(hdf5_file.get("features"))
            self.predicted_boxes = np.array(hdf5_file.get("bboxes"))
            self.indexes = np.array(hdf5_file.get("pos_bboxes"))

    def __getitem__(self, index):
        item = self.dataset[index]

        img_id = self.img_ids[index]
        feat_index = self.imgid2idx[int(img_id)]
        start_end_index = self.indexes[feat_index]
        num_box = min(start_end_index[1] - start_end_index[0], 36)
        # Get boxes
        boxes = np.zeros([36, 4]).astype(np.float32) - 1
        boxes[:num_box] = self.predicted_boxes[start_end_index[0] : start_end_index[1]][:num_box]
        # Get features
        obj_feats = np.zeros([36, 2048]).astype(np.float32)
        obj_feats[:num_box] = self.features[start_end_index[0] : start_end_index[1]][:num_box]
        # Get classes
        obj_cates = np.zeros([36]).astype(np.int32) - 1
        obj_cates[:num_box] = np.array([self.obj2ind.get(i) for i in
                              self.detection_dict[img_id]["classes"]]).astype(np.int32)[:num_box]

        return index, item, obj_feats, boxes, obj_cates

    def __len__(self):
        return len(self.dataset)
