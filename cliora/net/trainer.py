import os
import sys
import traceback
import types

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
# from torch_struct import TreeCRF
import numpy as np
import pickle
import cv2
# from cliora.net.cliora import DioraTreeLSTM, DioraMLP, DioraMLPShared
# from cliora.net.v_diora import DioraTreeLSTM, DioraMLP, DioraMLPShared
from cliora.logging.configuration import get_logger
from cliora.net.utils import ImageEncoder

COLOURS = [(255,0,0), (0,255,0), (165,42,42),(255,170,170), (255,255,255),
(0,127,255),(127,0,255),(127,255,0),(255,127,0),(255,0,127),(0,0,255),
(127,255,255), (255,127,255), (255,255,127), (127,255,127), (255,127,127),(127,127,255),
(127,0,63),(102,102,102),(64,192,192),(192,64,192),(192,192,64),(64,64,192),(64,192,64),(192,64,64)]


class ReconstructionSoftmaxLoss(nn.Module):
    name = 'reconstruct_softmax_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionSoftmaxLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin
        self.input_size = input_size

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        # self.mat_vis = nn.Parameter(torch.FloatTensor(size, input_size))
        self.lossfn = nn.CrossEntropyLoss()
        self._cuda = cuda
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        # input_size = self.input_size
        # size = cliora.outside_h.shape[-1]
        k = self.k_neg

        emb_pos = self.embeddings(sentences)
        emb_neg = self.embeddings(neg_samples.unsqueeze(0))

        # Calculate scores.

        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)
        proj_pos = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg = torch.matmul(emb_neg, torch.t(self.mat))
        # cell = cliora.outside_vh[:, :length].view(batch_size, length, 1, -1)
        # proj_pos = torch.matmul(emb_pos, torch.t(self.mat_vis))
        # proj_neg = torch.matmul(emb_neg, torch.t(self.mat_vis))

        ## The score.
        xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn = torch.einsum('zec,abxc->abe', proj_neg, cell)
        score = torch.cat([xp, xn], 2)

        # Calculate loss.
        inputs = score.view(batch_size * length, k + 1)
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)

        loss = self.lossfn(inputs, outputs)

        ret = dict(reconstruction_softmax_loss=loss)

        return loss, ret


class ContrastiveLoss(torch.nn.Module):
    name = 'contrastive_loss'

    def __init__(self, margin=1.0, alpha_contr=0.01, use_contr_ce=False):
        super(ContrastiveLoss, self).__init__()
        self.min_val = 1e-8
        self.margin = margin
        self.alpha_contr = alpha_contr
        self.use_contr_ce = use_contr_ce

    def forward(self, batch, diora):
        bs, seq_len = batch.shape
        inside_scores = diora.inside_s.squeeze(-1)  # bs*span_length*1
        outside_scores = diora.outside_s.squeeze(-1)
        # inside_scores = cliora.chart.inside_vs.squeeze(-1)  # bs*span_length*1
        # outside_scores = cliora.chart.outside_vs.squeeze(-1)
        span_length = inside_scores.shape[1]
        device = inside_scores.device

        # Flickr
        all_atten_score = diora.all_atten_score
        assert all_atten_score is not None
        scores = all_atten_score.max(-1).values # bs*bs*span_len
        # TODO: COCO
        # scores = cliora.all_atten_score # bs*bs*span_len

        scores = scores.permute(2, 0, 1)  # span_len*bs*bs
        diagonal = torch.diagonal(scores, 0, -1).unsqueeze(-1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.transpose(1, 2).expand_as(scores) # span_len*bs*bs

        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)
        I = (torch.eye(bs) > 0.5).unsqueeze(0).expand_as(scores).to(device)
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)

        loss_txt = loss_txt.mean(2) # span_len*bs
        loss_img = loss_img.mean(1) # span_len*bs

        vl_loss = (loss_txt + loss_img).t() # bs*span_len

        span_margs = torch.exp(inside_scores + outside_scores - inside_scores[:, [-1]]) # bs*span_length
        loss_mat = span_margs * vl_loss
        loss = loss_mat[:, :(span_length//2)].sum(-1).mean() * self.alpha_contr
        ret = dict(contrastive_loss=loss)

        return loss, ret


class VGLoss(torch.nn.Module):
    name = 'vg_loss'

    def __init__(self, alpha_vg=0.1):
        super(VGLoss, self).__init__()
        self.min_val = 1e-8
        self.alpha_vg = alpha_vg

    def forward(self, batch, vg_atten_score):
        # bs, seq_len = batch.shape

        batch_size, _, seq_len, _ = vg_atten_score.size()

        # [B, B, seq_len]
        phrase_region_max = vg_atten_score.max(-1).values

        """ V1 """
        phrase_region_scores = phrase_region_max.sum(-1)
        logits = phrase_region_scores.div(
            torch.tensor(seq_len, device=phrase_region_scores.device).expand(batch_size).unsqueeze(1).expand(
                phrase_region_scores.size())
        )

        """ V2 """
        # phrase_region_scores = phrase_region_max.sum(-1)
        # mask = phrase_region_scores.ge(0)
        # logits = phrase_region_scores.div(mask.sum(-1) + 1e-8)

        """ V3 """
        # mask = torch.softmax(phrase_region_max, -1)
        # phrase_region_scores = (phrase_region_max * mask).sum(-1)
        # logits = phrase_region_scores

        targets = torch.arange(
            batch_size, device=phrase_region_scores.device
        )


        loss = self.alpha_vg * F.cross_entropy(logits, targets)
        ret = dict(vg_loss=loss)
        return loss, ret


def get_loss_funcs(options, embedding_layer=None):
    input_dim = embedding_layer.weight.shape[1]
    size = options.hidden_dim
    k_neg = options.k_neg
    margin = options.margin
    vl_margin = options.vl_margin
    hinge_margin = options.hinge_margin
    alpha_contr = options.alpha_contr
    cuda = options.cuda

    loss_funcs = []

    # Reconstruction Loss
    reconstruction_loss_fn = ReconstructionSoftmaxLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)
    loss_funcs.append(reconstruction_loss_fn)

    # Visual Grounding Loss
    if options.vg_loss:
        vg_loss_fn = VGLoss(options.alpha_vg)
        loss_funcs.append(vg_loss_fn)

    # Contrastive Loss
    if options.obj_feats and options.use_contr:
        contrastive_loss_fn = ContrastiveLoss(vl_margin, alpha_contr, options.use_contr_ce)
        loss_funcs.append(contrastive_loss_fn)

    return loss_funcs


class Embed(nn.Module):
    def __init__(self, embeddings, input_size, size):
        super(Embed, self).__init__()
        self.input_size = input_size
        self.size = size
        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self.mat1 = nn.Parameter(torch.FloatTensor(size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, x):
        batch_size, length = x.shape
        emb = self.embeddings(x.view(-1))
        emb_span = torch.mm(emb, self.mat.t()).view(batch_size, length, -1)
        emb_word = torch.mm(emb, self.mat1.t()).view(batch_size, length, -1)
        return emb_span, emb_word


class Net(nn.Module):
    def __init__(self, embed, image_encoder, diora, obj_feats, visualize, loss_funcs=[]):
        super(Net, self).__init__()
        size = diora.size

        self.obj_feats = obj_feats
        if self.obj_feats:
            self.img_encoder = image_encoder
        self.embed = embed
        self.diora = diora
        self.visualize = visualize
        self.loss_func_names = [m.name for m in loss_funcs]

        for m in loss_funcs:
            setattr(self, m.name, m)

    def compute_loss(self, batch, neg_samples, info, batch_parse):
        ret, loss = {}, []

        # Loss
        diora = self.diora
        for func_name in self.loss_func_names:
            func = getattr(self, func_name)
            if 'reconstruct' in func_name:
                subloss, desc = func(batch, neg_samples, diora, info)
            elif 'contrastive' in func_name:
                subloss, desc = func(batch, diora)
            elif 'vg_loss' in func_name:
                subloss, desc = func(batch, diora.vg_atten_score)
                # subloss3, _ = func(batch, cliora.vg_atten_score3)
                # subloss2, _ = func(batch, cliora.vg_atten_score2)
                # subloss1, _ = func(batch, cliora.vg_atten_score1)
                # subloss = 0.1*(subloss1 + subloss2) + subloss3
                # # subloss = subloss2 + subloss3
                # desc = dict(vg_loss=subloss)
            else:
                continue
            loss.append(subloss.view(1, 1))
            for k, v in desc.items():
                ret[k] = v

        loss = torch.cat(loss, 1)

        return ret, loss

    def forward(self, img_ids, idx2word, batch, image_feats, obj_feats, boxes, obj_cates, neg_samples=None, compute_loss=True, info=None, batch_parse=None):
        """
        obj_cates: 1600 classes for flickr
        """
        # Embed
        # embed = self.embed(batch)
        embed_span, embed_word = self.embed(batch)

        img_embed = None; obj_embed_span = None; obj_embed_word = None
        if self.obj_feats:

            # img_embed, _ = self.img_encoder(image_feats, boxes, obj_cates) # TODO: COCO
            obj_embed_span, obj_embed_word = self.img_encoder(obj_feats=obj_feats) # Flickr


        # Run DIORA
        self.diora(embed_span, embed_word, obj_embed_span, obj_embed_word) # Flickr
        # self.cliora(embed, embed1, img_embed, obj_embed_v) # TODO: COCO

        # visualization
        if self.visualize:
            self.visualization(batch, img_ids, boxes, idx2word)

        # Compute Loss
        if compute_loss:
            ret, loss = self.compute_loss(batch, neg_samples, info=info, batch_parse=batch_parse)
        else:
            ret, loss = {}, torch.full((1, 1), 1, dtype=torch.float32, device=embed_span.device)

        # Results
        ret['total_loss'] = loss

        return ret


    def visualization(self, batch, img_ids, boxes, idx2word):
        max_prob, max_idx = self.diora.atten_score.max(-1)
        batch_size, length = batch.shape
        # img_root = './coco_data/'
        img_root = './flickr_data/'
        for bid, img_id in enumerate(img_ids):
            probs = max_prob[bid].cpu().tolist()
            # img_path = img_root + 'coco_images/val2014/COCO_val2014_' + str(img_id).zfill(12) + '.jpg'
            img_path = img_root + 'flickr30k_images/' + str(img_id) + '.jpg'
            img = cv2.imread(img_path)
            box_ids = max_idx[bid].cpu().tolist()
            box2color = {idx: i for i, idx in enumerate(set(box_ids))}
            if len(box2color) > len(COLOURS):
                continue
            sent_str = [idx2word[idx] for idx in batch[bid].cpu().tolist()]
            for l in range(length):
                box_id = box_ids[l]
                color_id = box2color[box_id]
                (x1, y1, x2, y2) = boxes[bid][box_id].cpu().tolist()
                # (x1, y1, x2, y2) = draw_boxes[l].cpu().tolist()
                img = cv2.rectangle(img, tuple([int(x1), int(y1)]), tuple([int(x2), int(y2)]),
                                    COLOURS[color_id], 2)
                # img = cv2.putText(img, str(l+1), tuple([int(x1+3*l), int(y1+5)]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #                   COLOURS[color_id], 2)
                img = cv2.putText(img, sent_str[l] + '   ' + str(round(probs[l], 2)), (10, 18 * (l + 1)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  COLOURS[color_id], 2)
            cv2.imwrite(img_root + 'visualize/{}.jpg'.format(img_id), img)


class Trainer(object):
    def __init__(self, net, k_neg=None, ngpus=None, cuda=None):
        super(Trainer, self).__init__()
        self.net = net
        self.optimizer = None
        self.optimizer_cls = None
        self.optimizer_kwargs = None
        self.cuda = cuda
        self.ngpus = ngpus

        self.parallel_model = None

        print("Trainer initialized with {} gpus.".format(ngpus))

    def freeze_diora(self):
        for p in self.net.diora.parameters():
            p.requires_grad = False

    def freeze_except_vis(self):
        for name, p in self.net.named_parameters():
            if '_vis' not in name:
                p.requires_grad = False

    def parameter_norm(self, requires_grad=True, diora=False):
        net = self.net.diora if diora else self.net
        total_norm = 0
        for p in net.parameters():
            if requires_grad and not p.requires_grad:
                continue
            total_norm += p.norm().item()
        return total_norm

    def init_optimizer(self, optimizer_cls, optimizer_kwargs):
        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls
        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

    @staticmethod
    def get_single_net(net):
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            return net.module
        return net

    def save_model(self, save_emb, model_file):
        state_dict = self.net.state_dict()

        if not save_emb:
            todelete = []

            for k in state_dict.keys():
                if 'embeddings' in k:
                    todelete.append(k)

            for k in todelete:
                del state_dict[k]

        torch.save({
            'state_dict': state_dict,
        }, model_file)

    @staticmethod
    def load_model(origin_emb, net, model_file):
        print('*'*6 + ' start initialization ' + '*'*6)
        save_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict_toload = save_dict['state_dict']
        state_dict_net = Trainer.get_single_net(net).state_dict()

        # Bug related to multi-gpu
        keys = list(state_dict_toload.keys())
        prefix = 'module.'
        for k in keys:
            if k.startswith(prefix):
                newk = k[len(prefix):]
                state_dict_toload[newk] = state_dict_toload[k]
                del state_dict_toload[k]

        # Remove extra keys.
        keys = list(state_dict_toload.keys())
        for k in keys:
            if k not in state_dict_net:
                print('deleting {}'.format(k))
                del state_dict_toload[k]

        # Hack to support embeddings.
        for k in state_dict_net.keys():
            if not origin_emb and 'embeddings' in k:
                state_dict_toload[k] = state_dict_net[k]
            elif k not in keys:
                if '_vis' in k and 'img_encoder' not in k:
                    state_dict_toload[k] = state_dict_toload[k.replace('_vis', '')]
                    continue
                print('Not initialize {}'.format(k))
                state_dict_toload[k] = state_dict_net[k]

        Trainer.get_single_net(net).load_state_dict(state_dict_toload)
        print('*' * 6 + ' end initialization ' + '*' * 6)

    def run_net(self, batch_map, idx2word=None, compute_loss=True, multigpu=False):
        img_ids = batch_map['example_ids']
        batch = batch_map['sentences']
        image_feats = batch_map['image_feats'].to(batch.device)
        neg_samples = batch_map.get('neg_samples', None)
        obj_feats = batch_map['obj_feats']
        boxes = batch_map['boxes']
        obj_cates = batch_map['obj_cates']
        batch_parse = batch_map['GT']
        info = self.prepare_info(batch_map)
        out = self.net(img_ids, idx2word, batch, image_feats, obj_feats, boxes, obj_cates, neg_samples=neg_samples, compute_loss=compute_loss, info=info, batch_parse=batch_parse)
        return out

    def gradient_update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        params = [p for p in self.net.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        self.optimizer.step()

    def prepare_result(self, batch_map, model_output):
        result = {}
        result['batch_size'] = batch_map['batch_size']
        result['length'] = batch_map['length']
        for k, v in model_output.items():
            if 'loss' in k:
                result[k] = v.mean(dim=0).sum().item()
        return result

    def prepare_info(self, batch_map):
        return {}

    def step(self, *args, **kwargs):
        try:
            return self._step(*args, **kwargs)
        except Exception as err:
            batch_map = args[0]
            print('Failed with shape: {}'.format(batch_map['sentences'].shape))
            if self.ngpus > 1:
                print(traceback.format_exc())
                print('The step failed. Running multigpu cleanup.')
                os.system("ps -elf | grep [p]ython | grep adrozdov | grep " + self.experiment_name + " | tr -s ' ' | cut -f 4 -d ' ' | xargs -I {} kill -9 {}")
                sys.exit()
            else:
                raise err

    def _step(self, batch_map, idx2word=None, train=True, compute_loss=True):
        if train:
            self.net.train()
        else:
            self.net.eval()
        multigpu = self.ngpus > 1 and train

        with torch.set_grad_enabled(train):
            model_output = self.run_net(batch_map, idx2word, compute_loss=compute_loss, multigpu=multigpu)

        # Calculate average loss for multi-gpu and sum for backprop.
        total_loss = model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss)

        result = self.prepare_result(batch_map, model_output)

        return result


def build_net(options, embeddings=None, random_seed=None):

    logger = get_logger()

    lr = options.lr
    size = options.hidden_dim
    k_neg = options.k_neg
    # margin = options.margin
    normalize = options.normalize
    cuda = options.cuda
    rank = options.local_rank
    share = options.share
    ngpus = 1

    if options.arch == 'mlp':
        if options.obj_feats:
            # from cliora.net.vg import DioraMLP as Diora
            # from cliora.net.diora import DioraMLP as Diora
            from cliora.net.cliora import DioraMLP as Diora
        else:
            from cliora.net.diora import DioraMLP as Diora
    else:
        raise NotImplementedError

    if cuda and options.multigpu:
        ngpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = options.master_addr
        os.environ['MASTER_PORT'] = options.master_port
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Embed
    origin_emb = options.emb == 'none'
    if origin_emb:
        embedding_layer = embeddings
        if options.obj_feats:
            # When load pretrained DIORA to finetune CLIORA, we need to freeze
            # word embedding layer to reduce memory costs
            embedding_layer.weight.requires_grad = False
    else:
        if options.emb == 'skip':
            embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True, padding_idx=0)
        else:
            embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
    embed = Embed(embedding_layer, input_size=embedding_layer.weight.size(1), size=size)

    image_encoder = ImageEncoder(input_size=2048, size=size)

    # Diorav
    diora = Diora(size, outside=True, normalize=normalize, compress=False, share=share)

    # Loss
    loss_funcs = get_loss_funcs(options, embedding_layer)

    # Net
    net = Net(embed, image_encoder, diora, obj_feats=options.obj_feats, visualize=options.visualize, loss_funcs=loss_funcs)

    # Load model.
    if options.load_model_path is not None:
        logger.info('Loading model: {}'.format(options.load_model_path))
        Trainer.load_model(origin_emb, net, options.load_model_path)

    # CUDA-support
    if cuda:
        if options.multigpu:
            torch.cuda.set_device(options.local_rank)
        net.cuda()
        diora.cuda()

    if cuda and options.multigpu:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Trainer
    trainer = Trainer(net, k_neg=k_neg, ngpus=ngpus, cuda=cuda)
    trainer.rank = rank
    trainer.experiment_name = options.experiment_name # for multigpu cleanup
    trainer.init_optimizer(optim.Adam, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

    return trainer
