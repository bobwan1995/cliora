import torch
import torch.nn as nn

from cliora.net.utils import *

class Chart(object):
    def __init__(self, batch_size, length, size, dtype=None, cuda=False):
        super(Chart, self).__init__()

        ncells = int(length * (1 + length) / 2)

        device = torch.cuda.current_device() if cuda else None

        ## Inside.
        self.inside_h = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
        self.inside_c = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
        self.inside_s = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)

        ## Outside.
        self.outside_h = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
        self.outside_c = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
        self.outside_s = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)

        ## Visual
        self.vis_aggragate = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)


class AttentionHead(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, h_dim):
        super(AttentionHead, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(0.1)


    def forward(self, h_q, h_k, h_v, temp=1.0):

        all_atten_score = torch.einsum('abx,cdx->acbd', h_q, h_k)
        # atten_score = torch.diagonal(all_atten_score, 0, 0, 1).permute(2, 0, 1) / self.h_dim**0.5
        atten_score = torch.diagonal(all_atten_score/temp, 0, 0, 1).permute(2, 0, 1)
        atten_prob = self.dropout(torch.softmax(atten_score, dim=-1))
        cxt = torch.bmm(atten_prob, h_v)
        return cxt


# Composition Functions
class VLComposeMLP(nn.Module):
    def __init__(self, size, ninput=2, leaf=False):
        super(VLComposeMLP, self).__init__()

        self.size = size
        self.ninput = ninput
        if leaf:
            self.leaf_fc = nn.Linear(self.size, self.size)
        self.h_fcs = nn.Sequential(
            nn.Linear(2 * self.size, self.size),
            nn.ReLU(),
            nn.Linear(self.size, self.size),
            nn.ReLU()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0


    def leaf_transform(self, x, obj, atten_head, normalize_func):
        h = torch.tanh(self.leaf_fc(x))
        h = normalize_func(h) # TODO

        cxt = atten_head(h, obj, obj)
        # cxt = obj.mean(1).unsqueeze(1).expand(h.shape)

        # visual as residule
        h = h + cxt
        return h, cxt

    def forward(self, hs, cs):
        input_h = torch.cat(hs, 1)
        h = self.h_fcs(input_h)
        c = torch.full(h.shape, 0, dtype=torch.float32, device=h.device)
        return h, c


# Score Functions

class Bilinear(nn.Module):
    def __init__(self, size):
        super(Bilinear, self).__init__()
        self.size = size
        self.mat = nn.Parameter(torch.FloatTensor(self.size, self.size))

    def forward(self, vector1, vector2):
        bma = torch.matmul(vector1, self.mat).unsqueeze(1)
        ba = torch.matmul(bma, vector2.unsqueeze(2)).view(-1, 1)
        return ba


# Inside

def inside_fill_chart(batch_info, chart, index, h, c, s):
    L = batch_info.length - batch_info.level

    offset = index.get_offset(batch_info.length)[batch_info.level]

    chart.inside_h[:, offset:offset+L] = h
    chart.inside_c[:, offset:offset+L] = c
    chart.inside_s[:, offset:offset+L] = s


def get_inside_states(batch_info, chart, index, size):
    lidx, ridx = index.get_inside_index(batch_info.length, batch_info.level)

    ls = chart.index_select(index=lidx, dim=1).view(-1, size)
    rs = chart.index_select(index=ridx, dim=1).view(-1, size)

    return ls, rs


def inside_compose(compose_func, hs, cs):
    return compose_func(hs, cs)


def inside_score(score_func, batch_info, hs, cs, ss):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level
    N = batch_info.level

    s = score_func(hs[0], hs[1]) + ss[0] + ss[1]
    s = s.view(B, L, N, 1)
    p = torch.softmax(s, dim=2)

    return s, p


def inside_aggregate(batch_info, h, c, s, p, obj, normalize_func, atten_head):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level
    N = batch_info.level

    h_agg = torch.sum(h.view(B, L, N, -1) * p, 2)
    c_agg = torch.sum(c.view(B, L, N, -1) * p, 2)
    s_agg = torch.sum(s * p, 2)

    h_agg = normalize_func(h_agg) # TODO
    # cxt = obj.mean(1).unsqueeze(1).expand(h_agg.shape)
    cxt = atten_head(h_agg, obj, obj)
    h_agg = h_agg + cxt

    h_agg = normalize_func(h_agg)
    c_agg = normalize_func(c_agg) # ignore

    return h_agg, c_agg, s_agg


# Outside

def outside_fill_chart(batch_info, chart, index, h, c, s):
    L = batch_info.length - batch_info.level

    offset = index.get_offset(batch_info.length)[batch_info.level]

    chart.outside_h[:, offset:offset+L] = h
    chart.outside_c[:, offset:offset+L] = c
    chart.outside_s[:, offset:offset+L] = s


def get_outside_states(batch_info, pchart, schart, index, size):
    pidx, sidx = index.get_outside_index(batch_info.length, batch_info.level)

    ps = pchart.index_select(index=pidx, dim=1).view(-1, size)
    ss = schart.index_select(index=sidx, dim=1).view(-1, size)

    return ps, ss


def outside_compose(compose_func, hs, cs):
    return compose_func(hs, cs)


def outside_score(score_func, batch_info, hs, cs, ss):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level

    s = score_func(hs[0], hs[1]) + ss[0] + ss[1]
    s = s.view(B, -1, L, 1)
    p = torch.softmax(s, dim=1)

    return s, p


def outside_aggregate(batch_info, h, c, s, p, normalize_func):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level
    N = s.shape[1]

    h_agg = torch.sum(h.view(B, N, L, -1) * p, 1)
    c_agg = torch.sum(c.view(B, N, L, -1) * p, 1)
    s_agg = torch.sum(s * p, 1)

    h_agg = normalize_func(h_agg)
    c_agg = normalize_func(c_agg)

    return h_agg, c_agg, s_agg


# Base

class DioraBase(nn.Module):
    r"""DioraBase

    """

    def __init__(self, size, outside=True, normalize='unit', compress=False, share=True):
        super(DioraBase, self).__init__()
        assert normalize in ('none', 'unit'), 'Does not support "{}".'.format(normalize)

        self.share = share
        self.size = size
        self.outside = outside
        self.inside_normalize_func = NormalizeFunc(normalize)
        self.outside_normalize_func = NormalizeFunc(normalize)
        self.compress = compress
        self.atten_head = AttentionHead(size, size, size, size)
        self.ninput = 2

        self.index = None
        self.charts = None

        self.init_parameters()
        self.reset_parameters()
        self.reset()

    def init_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def inside_h(self):
        return self.chart.inside_h

    @property
    def inside_c(self):
        return self.chart.inside_c

    @property
    def inside_s(self):
        return self.chart.inside_s

    @property
    def outside_h(self):
        return self.chart.outside_h

    @property
    def outside_c(self):
        return self.chart.outside_c

    @property
    def outside_s(self):
        return self.chart.outside_s

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0

    def cuda(self):
        super(DioraBase, self).cuda()
        if self.index is not None:
            self.index.cuda = True # TODO: Should support to/from cpu/gpu.

    def get(self, chart, level):
        length = self.length
        L = length - level
        offset = self.index.get_offset(length)[level]
        return chart[:, offset:offset+L]

    def leaf_transform(self, x, obj_embed):
        normalize_func = self.inside_normalize_func
        transform_func = self.inside_compose_func.leaf_transform
        atten_head = self.atten_head

        input_shape = x.shape[:-1]
        h, c = transform_func(x, obj_embed, atten_head, normalize_func)

        h = normalize_func(h.view(*input_shape, self.size))
        c = normalize_func(c.view(*input_shape, self.size)) # ignore

        return h, c

    # Inside
    def inside_func(self, compose_func, score_func, atten_func, obj_embed, batch_info, chart, index, normalize_func):
        lh, rh = get_inside_states(batch_info, chart.inside_h, index, batch_info.size)
        lc, rc = get_inside_states(batch_info, chart.inside_c, index, batch_info.size)
        ls, rs = get_inside_states(batch_info, chart.inside_s, index, 1)

        hlst = [lh, rh]
        clst = [lc, rc]
        slst = [ls, rs]

        h, c = inside_compose(compose_func, hlst, clst)
        s, p = inside_score(score_func, batch_info, hlst, clst, slst)
        hbar, cbar, sbar = inside_aggregate(batch_info, h, c, s, p, obj_embed, normalize_func, atten_func)

        inside_fill_chart(batch_info, chart, index, hbar, cbar, sbar)

        return h, c, s

    def inside_pass(self, obj_embed):
        compose_func = self.inside_compose_func
        score_func = self.inside_score_func
        index = self.index
        chart = self.chart
        normalize_func = self.inside_normalize_func
        atten_func = self.atten_head

        for level in range(1, self.length):

            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
            )

            h, c, s = self.inside_func(compose_func, score_func, atten_func, obj_embed,
                                  batch_info, chart, index, normalize_func=normalize_func)

            self.inside_hook(level, h, c, s)


    def inside_hook(self, level, h, c, s):
        pass

    def outside_hook(self, level, h, c, s):
        pass

    def initialize_outside_root(self):
        B = self.batch_size
        D = self.size
        normalize_func = self.outside_normalize_func

        if self.compress:
            h = torch.matmul(self.inside_h[:, -1:], self.root_mat_out)
        else:
            h = self.root_vector_out_h.view(1, 1, D).expand(B, 1, D)
        if self.root_vector_out_c is None:
            device = torch.cuda.current_device() if self.is_cuda else None
            c = torch.full(h.shape, 0, dtype=torch.float32, device=device)
        else:
            c = self.root_vector_out_c.view(1, 1, D).expand(B, 1, D)

        h = normalize_func(h)
        c = normalize_func(c)

        self.chart.outside_h[:, -1:] = h
        self.chart.outside_c[:, -1:] = c


    def outside_func(self, compose_func, score_func, batch_info, chart, index, normalize_func):
        ph, sh = get_outside_states(
            batch_info, chart.outside_h, chart.inside_h, index, batch_info.size)
        pc, sc = get_outside_states(
            batch_info, chart.outside_c, chart.inside_c, index, batch_info.size)
        ps, ss = get_outside_states(
            batch_info, chart.outside_s, chart.inside_s, index, 1)

        hlst = [sh, ph]
        clst = [sc, pc]
        slst = [ss, ps]

        h, c = outside_compose(compose_func, hlst, clst)
        s, p = outside_score(score_func, batch_info, hlst, clst, slst)
        hbar, cbar, sbar = outside_aggregate(batch_info, h, c, s, p, normalize_func)

        # TODO: add attention here
        outside_fill_chart(batch_info, chart, index, hbar, cbar, sbar)

        return h, c, s


    def outside_pass(self):
        self.initialize_outside_root()

        compose_func = self.outside_compose_func
        score_func = self.outside_score_func
        index = self.index
        chart = self.chart
        normalize_func = self.outside_normalize_func

        for level in range(self.length - 2, -1, -1):
            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
                )

            h, c, s = self.outside_func(compose_func, score_func,
                                   batch_info, chart, index, normalize_func=normalize_func)

            self.outside_hook(level, h, c, s)

    # Initialization
    def init_with_batch(self, h, c):
        size = self.size
        batch_size, length, _ = h.shape

        self.batch_size = batch_size
        self.length = length

        self.chart = Chart(batch_size, length, size, dtype=torch.float32, cuda=self.is_cuda)
        self.chart.inside_h[:, :self.length] = h
        self.chart.inside_c[:, :self.length] = c

    def reset(self):
        self.batch_size = None
        self.length = None
        self.chart = None
        self.all_atten_score = None
        self.atten_score = None

    def get_chart_wrapper(self):
        return self

    def forward(self, x_span, x_word, obj_embed_span=None, obj_embed_word=None):
        if self.index is None:
            self.index = Index(cuda=self.is_cuda)

        self.reset()

        h, c = self.leaf_transform(x_span, obj_embed_span)

        self.init_with_batch(h, c)

        self.inside_pass(obj_embed_span)

        if self.outside:
            self.outside_pass()

        # TODO: COCO
        # self.all_atten_score = orch.einsum('abx,cx->acb', self.chart.inside_h + self.chart.outside_h, obj_embed_span)

        # Flickr
        self.all_atten_score = torch.einsum('abx,cdx->acbd', self.chart.inside_h + self.chart.outside_h, obj_embed_span)

        if self.training:
            self.vg_atten_score_word = torch.einsum('abx,cdx->acbd', x_word, obj_embed_word)
            self.vg_atten_score = self.vg_atten_score_word
        else:
            self.vg_atten_score_word = torch.einsum('abx,cdx->acbd', self.inside_normalize_func(x_word), obj_embed_word)
            self.vg_atten_score = self.all_atten_score[:, :, :x_span.size(1)] + self.vg_atten_score_word

        self.atten_score = torch.diagonal(self.vg_atten_score, 0, 0, 1).permute(2, 0, 1)

        return None


class DioraMLP(DioraBase):

    def init_parameters(self):
        self.inside_score_func = Bilinear(self.size)
        self.inside_compose_func = VLComposeMLP(self.size, leaf=True)
        if self.share:
            self.outside_score_func = self.inside_score_func
            self.outside_compose_func = self.inside_compose_func
        else:
            self.outside_score_func = Bilinear(self.size)
            self.outside_compose_func = VLComposeMLP(self.size)

        if self.compress:
            self.root_mat_out = nn.Parameter(torch.FloatTensor(self.size, self.size))
        else:
            self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))

        self.root_vector_out_c = None
