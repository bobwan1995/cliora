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


# Composition Functions

# class TreeLSTM(nn.Module):
#     def __init__(self, size, ninput=2, leaf=False):
#         super(TreeLSTM, self).__init__()
#
#         self.size = size
#         self.ninput = ninput
#
#         if leaf:
#             self.W = nn.Parameter(torch.FloatTensor(3 * self.size, self.size))
#         self.U = nn.Parameter(torch.FloatTensor(5 * self.size, self.ninput * self.size))
#         self.B = nn.Parameter(torch.FloatTensor(5 * self.size))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         params = [p for p in self.parameters() if p.requires_grad]
#         for i, param in enumerate(params):
#             param.data.normal_()
#
#     def leaf_transform(self, x):
#         W, B = self.W, self.B[:3*self.size]
#
#         activations = torch.matmul(x, W.t()) + B
#         a_lst = torch.chunk(activations, 3, dim=-1)
#         u = torch.tanh(a_lst[0])
#         i = torch.sigmoid(a_lst[1])
#         o = torch.sigmoid(a_lst[2])
#
#         c = i * u
#         h = o * torch.tanh(c)
#
#         return h, c
#
#     def forward(self, hs, cs, constant=1.0):
#         U, B = self.U, self.B
#
#         input_h = torch.cat(hs, 1)
#
#         activations = torch.matmul(input_h, U.t()) + B
#         a_lst = torch.chunk(activations, 5, dim=1)
#         u = torch.tanh(a_lst[0])
#         i = torch.sigmoid(a_lst[1])
#         o = torch.sigmoid(a_lst[2])
#         f0 = torch.sigmoid(a_lst[3] + constant)
#         f1 = torch.sigmoid(a_lst[4] + constant)
#
#         c = f0 * cs[0] + f1 * cs[1] + i * u
#         h = o * torch.tanh(c)
#
#         return h, c


class ComposeMLP(nn.Module):
    def __init__(self, size, ninput=2, leaf=False):
        super(ComposeMLP, self).__init__()

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
        # self.reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0

    # def reset_parameters(self):
    #     # TODO: Init with diagonal.
    #     params = [p for p in self.parameters() if p.requires_grad]
    #     for i, param in enumerate(params):
    #         param.data.normal_()

    def leaf_transform(self, x):
        # h = self.leaf_fc(x)
        h = torch.tanh(self.leaf_fc(x))
        c = torch.full(h.shape, 0, dtype=torch.float32, device=h.device)

        return h, c

    def forward(self, hs, cs=None, constant=1.0):
        input_h = torch.cat(hs, 1)
        h = self.h_fcs(input_h)

        # device = torch.cuda.current_device() if self.is_cuda else None
        c = torch.full(h.shape, 0, dtype=torch.float32, device=h.device)

        return h, c


# Score Functions

class Bilinear(nn.Module):
    def __init__(self, size):
        super(Bilinear, self).__init__()
        self.size = size
        self.mat = nn.Parameter(torch.FloatTensor(self.size, self.size))
        # self.reset_parameters()

    # def reset_parameters(self):
    #     params = [p for p in self.parameters() if p.requires_grad]
    #     for i, param in enumerate(params):
    #         param.data.normal_()

    def forward(self, vector1, vector2):
        # bilinear
        # a = 1 (in a more general bilinear function, a is any positive integer)
        # vector1.shape = (b, m)
        # matrix.shape = (m, n)
        # vector2.shape = (b, n)
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


def inside_score(score_func, batch_info, hs, ss):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level
    N = batch_info.level

    s = score_func(hs[0], hs[1]) + ss[0] + ss[1]
    s = s.view(B, L, N, 1)
    p = torch.softmax(s, dim=2)

    return s, p


def inside_aggregate(batch_info, h, c, s, p, normalize_func):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level
    N = batch_info.level

    h_agg = torch.sum(h.view(B, L, N, -1) * p, 2)
    c_agg = torch.sum(c.view(B, L, N, -1) * p, 2)
    s_agg = torch.sum(s * p, 2)

    h_agg = normalize_func(h_agg)
    c_agg = normalize_func(c_agg)

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
    return compose_func(hs, cs, 0)


def outside_score(score_func, batch_info, hs, ss):
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

    def __init__(self, size, word_mat=None, cate_mat=None, outside=True, normalize='unit', compress=False, share=True):
        super(DioraBase, self).__init__()
        assert normalize in ('none', 'unit', 'xavier'), 'Does not support "{}".'.format(normalize)

        self.size = size
        self.share = share
        self.outside = outside
        self.inside_normalize_func = NormalizeFunc(normalize)
        self.outside_normalize_func = NormalizeFunc(normalize)
        # self.inside_normalize_func = nn.LayerNorm(size)
        # self.outside_normalize_func = nn.LayerNorm(size)
        self.compress = compress
        self.ninput = 2

        self.index = None
        self.charts = None

        self.init_parameters()
        self.reset_parameters() # reset all submodules
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

    def leaf_transform(self, x):
        normalize_func = self.inside_normalize_func
        transform_func = self.inside_compose_func.leaf_transform

        input_shape = x.shape[:-1]
        h, c = transform_func(x)
        h = normalize_func(h.view(*input_shape, self.size))
        c = normalize_func(c.view(*input_shape, self.size))

        return h, c

    # Inside
    def inside_func(self, compose_func, score_func, batch_info, chart, index, normalize_func):
        lh, rh = get_inside_states(batch_info, chart.inside_h, index, batch_info.size)
        lc, rc = get_inside_states(batch_info, chart.inside_c, index, batch_info.size)
        ls, rs = get_inside_states(batch_info, chart.inside_s, index, 1)

        hlst = [lh, rh]
        clst = [lc, rc]
        slst = [ls, rs]

        h, c = inside_compose(compose_func, hlst, clst)
        s, p = inside_score(score_func, batch_info, hlst, slst)
        hbar, cbar, sbar = inside_aggregate(batch_info, h, c, s, p, normalize_func)

        inside_fill_chart(batch_info, chart, index, hbar, cbar, sbar)

        return h, c, s

    def inside_pass(self):
        compose_func = self.inside_compose_func
        score_func = self.inside_score_func
        index = self.index
        chart = self.chart
        normalize_func = self.inside_normalize_func

        for level in range(1, self.length):

            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
                )

            h, c, s = self.inside_func(compose_func, score_func, batch_info, chart, index,
                normalize_func=normalize_func)

            self.inside_hook(level, h, c, s)

    def inside_hook(self, level, h, c, s):
        pass

    # Outside
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
        s, p = outside_score(score_func, batch_info, hlst, slst)
        hbar, cbar, sbar = outside_aggregate(batch_info, h, c, s, p, normalize_func)

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

            h, c, s = self.outside_func(compose_func, score_func, batch_info, chart, index,
                normalize_func=normalize_func)

            self.outside_hook(level, h, c, s)

    def outside_hook(self, level, h, c, s):
        pass

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
        self.atten_score = None
        self.all_atten_score = None
        self.vg_atten_score = None


    def forward(self, x, obj_embed=None, obj_embed_v=None, prior_sim=None):

        self.vg_atten_score = torch.einsum('abx,cdx->acbd', x, obj_embed)
        self.atten_score = torch.diagonal(self.vg_atten_score, 0, 0, 1).permute(2, 0, 1)

        return None


class DioraMLP(DioraBase):

    def init_parameters(self):
        self.inside_score_func = Bilinear(self.size)
        self.inside_compose_func = ComposeMLP(self.size, leaf=True)

        if self.share:
            self.outside_score_func = self.inside_score_func
            self.outside_compose_func = self.inside_compose_func
        else:
            self.outside_score_func = Bilinear(self.size)
            self.outside_compose_func = ComposeMLP(self.size)

        if self.compress:
            self.root_mat_out = nn.Parameter(torch.FloatTensor(self.size, self.size))
        else:
            self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))
        self.root_vector_out_c = None
        # self.root_vector_out_c = nn.Parameter(torch.FloatTensor(self.size))
