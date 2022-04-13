import torch
import torch.nn as nn
from scipy.special import factorial

from cliora.net.outside_index import get_outside_index, get_topk_outside_index, get_outside_components
from cliora.net.inside_index import get_inside_index, get_inside_components, get_inside_index_unique
from cliora.net.offset_cache import get_offset_cache

from collections import OrderedDict

TINY = 1e-8
class UnitNorm(object):
    def __call__(self, x, p=2, eps=TINY):
        return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


class NormalizeFunc(nn.Module):
    def __init__(self, mode='none'):
        super(NormalizeFunc, self).__init__()
        self.mode = mode

    def forward(self, x):
        mode = self.mode
        if mode == 'none':
            return x
        elif mode == 'unit':
            return UnitNorm()(x)


class BatchInfo(object):
    def __init__(self, **kwargs):
        super(BatchInfo, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class ImageEncoder(nn.Module):
    def __init__(self, input_size, size):
        super(ImageEncoder, self).__init__()

        self.fc = nn.Linear(input_size, size)
        self.fc_vis = nn.Linear(input_size, size)
        self.reset_parameters()

    def reset_parameters(self):
        # zero norm keep same with MAF
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            # param.data.normal_()
            param.data.zero_()

    def forward(self, obj_feats):
        features_span = self.fc(obj_feats.float())
        features_word = self.fc_vis(obj_feats.float())
        return features_span, features_word


def get_catalan(n):
    if n > 10:
        return 5000 # HACK: We only use this to check number of trees, and this avoids overflow.
    n = n - 1
    def choose(n, p):
        return factorial(n) / (factorial(p) * factorial(n-p))
    return int(choose(2 * n, n) // (n + 1))


class Index(object):
    def __init__(self, cuda=False, enable_caching=True):
        super(Index, self).__init__()
        self.cuda = cuda
        self.cache = {}
        self.inside_index_cache = {}
        self.inside_index_unique_cache = {}
        self.outside_index_cache = {}
        self.outside_encoded_index_cache = {}
        self.offset_cache = {}
        self.enable_caching = enable_caching

    def cached_lookup(self, func, name, key):
        if name not in self.cache:
            self.cache[name] = {}
        cache = self.cache[name]
        if self.enable_caching:
            if key not in cache:
                cache[key] = func()
            return cache[key]
        else:
            return func()

    def get_catalan(self, n):
        name = 'catalan'
        key = n
        def func():
            return get_catalan(n)
        return self.cached_lookup(func, name, key)

    def get_offset(self, length):
        name = 'offset_cache'
        key = length
        def func():
            return get_offset_cache(length)
        return self.cached_lookup(func, name, key)

    def get_inside_index(self, length, level):
        name = 'inside_index_cache'
        key = (length, level)
        def func():
            return get_inside_index(length, level,
                self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)

    def get_inside_index_unique(self, length, level):
        name = 'inside_index_unique_cache'
        key = (length, level)
        def func():
            return get_inside_index_unique(length, level,
                self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)

    def get_outside_index(self, length, level):
        name = 'outside_index_cache'
        key = (length, level)
        def func():
            return get_outside_index(length, level,
                self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)

    def get_topk_outside_index(self, length, level, K):
        name = 'topk_outside_index_cache'
        key = (length, level, K)
        def func():
            return get_topk_outside_index(length, level, K,
                self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)