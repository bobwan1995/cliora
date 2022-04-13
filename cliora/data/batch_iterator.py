from cliora.data.dataloader import FixedLengthBatchSampler, SimpleDataset, COCODataset, FlickrDataset
from cliora.blocks.negative_sampler import choose_negative_samples

import torch
import numpy as np

def get_config(config, **kwargs):
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
    return config


def get_default_config():

    default_config = dict(
        batch_size=16,
        forever=False,
        drop_last=False,
        sort_by_length=True,
        shuffle=True,
        random_seed=None,
        filter_length=None,
        workers=16,
        pin_memory=False,
        include_partial=False,
        cuda=False,
        ngpus=1,
        k_neg=3,
        negative_sampler=None,
        options_path=None,
        weights_path=None,
        vocab=None,
        length_to_size=None,
        rank=None,
        data_type=None,
        use_obj=False,
        mode=None,
    )

    return default_config


class BatchIterator(object):

    def __init__(self, sentences, extra={}, **kwargs):
        self.sentences = sentences
        self.config = get_config(get_default_config(), **kwargs)
        self.extra = extra
        self.loader = None
        self.get_dataset()

    def chunk(self, tensor, chunks, dim=0, i=0):
        if isinstance(tensor, torch.Tensor):
            return torch.chunk(tensor, chunks, dim=dim)[i]
        index = torch.chunk(torch.arange(len(tensor)), chunks, dim=dim)[i]
        return [tensor[ii] for ii in index]

    def partition(self, tensor, rank, device_ids):
        if tensor is None:
            return None
        if isinstance(tensor, dict):
            for k, v in tensor.items():
                tensor[k] = self.partition(v, rank, device_ids)
            return tensor
        return self.chunk(tensor, len(device_ids), 0, rank)

    def get_dataset_size(self):
        return len(self.sentences)

    def get_dataset_minlen(self):
        return min(map(len, self.sentences))

    def get_dataset_maxlen(self):
        return max(map(len, self.sentences))

    def get_dataset_stats(self):
        return 'size={} minlen={} maxlen={}'.format(
            self.get_dataset_size(), self.get_dataset_minlen(), self.get_dataset_maxlen()
        )

    def choose_negative_samples(self, negative_sampler, k_neg):
        return choose_negative_samples(negative_sampler, k_neg)

    def get_dataset(self):
        data_type = self.config.get('data_type')
        use_obj = self.config.get('use_obj')
        mode = self.config.get('mode')
        if use_obj and data_type == 'coco':
            dataset = COCODataset(self.sentences, self.extra['example_ids'])
        elif use_obj and data_type == 'flickr':
            dataset = FlickrDataset(self.sentences, self.extra['example_ids'], mode)
        else:
            dataset = SimpleDataset(self.sentences)
        self.dataset = dataset

    def get_iterator(self, **kwargs):
        config = get_config(self.config.copy(), **kwargs)

        random_seed = config.get('random_seed')
        batch_size = config.get('batch_size')
        filter_length = config.get('filter_length')
        pin_memory = config.get('pin_memory')
        include_partial = config.get('include_partial')
        cuda = config.get('cuda')
        ngpus = config.get('ngpus')
        rank = config.get('rank')
        k_neg = config.get('k_neg')
        negative_sampler = config.get('negative_sampler', None)
        workers = config.get('workers')
        length_to_size = config.get('length_to_size', None)
        # data_type = config.get('data_type')
        # use_obj = config.get('use_obj')
        # debug = config.get('debug')

        def collate_fn(batch):
            index, sents, obj_feats, boxes, obj_cates = zip(*batch)
            sents = torch.from_numpy(np.array(sents)).long()
            obj_feats = torch.from_numpy(np.array(obj_feats))
            boxes = torch.from_numpy(np.array(boxes))
            obj_cates = torch.from_numpy(np.array(obj_cates)).long()

            batch_map = {}
            batch_map['index'] = index
            batch_map['sents'] = sents
            batch_map['obj_feats'] = obj_feats
            batch_map['boxes'] = boxes
            batch_map['obj_cates'] = obj_cates

            for k, v in self.extra.items():
                batch_map[k] = [v[idx] for idx in index]
            batch_map['image_feats'] = torch.from_numpy(np.array(batch_map['image_feats']))

            if ngpus > 1:
                for k in batch_map.keys():
                    batch_map[k] = self.partition(batch_map[k], rank, range(ngpus))

            return batch_map

        if self.loader is None:
            rng = np.random.RandomState(seed=random_seed)
            sampler = FixedLengthBatchSampler(self.dataset, batch_size=batch_size, rng=rng,
                maxlen=filter_length, include_partial=include_partial, length_to_size=length_to_size)
            loader = torch.utils.data.DataLoader(self.dataset, shuffle=(sampler is None), num_workers=workers, pin_memory=pin_memory,batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader

        def myiterator():

            for batch in self.loader:
                index = batch['index']
                sentences = batch['sents']
                obj_feats = batch['obj_feats']
                boxes = batch['boxes']
                obj_cates = batch['obj_cates']

                batch_size, length = sentences.shape

                neg_samples = None
                if negative_sampler is not None:
                    neg_samples = self.choose_negative_samples(negative_sampler, k_neg)

                if cuda:
                    sentences = sentences.cuda()
                    obj_feats = obj_feats.cuda()
                    boxes = boxes.cuda()
                    obj_cates = obj_cates.cuda()
                if cuda and neg_samples is not None:
                    neg_samples = neg_samples.cuda()

                batch_map = {}
                batch_map['sentences'] = sentences
                batch_map['neg_samples'] = neg_samples
                batch_map['batch_size'] = batch_size
                batch_map['length'] = length
                batch_map['obj_feats'] = obj_feats
                batch_map['boxes'] = boxes
                batch_map['obj_cates'] = obj_cates

                for k, v in self.extra.items():
                    batch_map[k] = batch[k]

                yield batch_map

        return myiterator()

