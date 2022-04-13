import os
import collections
import json

import torch
import torchvision.ops as torchops
import numpy as np
from tqdm import tqdm

from train import argument_parser, parse_args, configure
from train import get_validation_dataset, get_validation_iterator
from train import build_net

from cliora.logging.configuration import get_logger

from cliora.analysis.cky import ParsePredictor as CKY
from cliora.analysis.utils import *
import copy

punctuation_words = set([x.lower() for x in ['.', ',', ':', '-LRB-', '-RRB-', '\'\'',
    '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']])


def remove_using_flat_mask(tr, mask):
    kept, removed = [], []
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        node = []

        for subtree in tr:
            x, xsize = func(subtree, pos=pos + size)
            if x is not None:
                node.append(x)
            size += xsize

        if len(node) == 1:
            node = node[0]
        elif len(node) == 0:
            return None, size
        return node, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)


def postprocess(tr, tokens=None):
    if tokens is None:
        tokens = flatten_tree(tr)

    # Don't remove the last token. It's not punctuation.
    if tokens[-1].lower() not in punctuation_words:
        return tr

    mask = [True] * (len(tokens) - 1) + [False]
    tr, kept, removed = remove_using_flat_mask(tr, mask)
    assert len(kept) == len(tokens) - 1, 'Incorrect tokens left. Output = {}, Kept = {}'.format(
        tr, kept)
    assert len(kept) > 0, 'No tokens left. Original = {}'.format(tokens)
    assert len(removed) == 1
    tr = (tr, tokens[-1])

    return tr


def replace_leaves(tree, leaves):
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            return 1, leaves[pos]

        newtree = []
        sofar = 0
        for node in tr:
            size, newnode = func(node, pos+sofar)
            sofar += size
            newtree += [newnode]

        return sofar, newtree

    _, newtree = func(tree)

    return newtree


def run(options):
    logger = get_logger()

    validation_dataset = get_validation_dataset(options)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    word2idx = validation_dataset['word2idx']
    embeddings = validation_dataset['embeddings']

    idx2word = {v: k for k, v in word2idx.items()}

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings)

    # Parse

    diora = trainer.net.diora

    ## Monkey patch parsing specific methods.
    override_init_with_batch(diora)
    override_inside_hook(diora)

    ## Turn off outside pass.
    trainer.net.diora.outside = True
    # trainer.net.cliora.outside = False

    ## Eval mode.
    trainer.net.eval()

    ## Parse predictor.
    parse_predictor = CKY(net=diora)

    batches = validation_iterator.get_iterator(random_seed=options.seed)

    output_path = os.path.abspath(os.path.join(options.experiment_path, 'parse.jsonl'))

    logger.info('Beginning.')
    logger.info('Writing output to = {}'.format(output_path))

    total_num = 0.
    recall_num = 0.
    ccr_num = 0.
    corpus_f1 = [0., 0., 0.]
    sent_f1 = []
    f = open(output_path, 'w')

    recon_loss = 0
    vg_loss = 0
    contr_loss = 0
    total_loss = 0
    num_data = 0

    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            sentences = batch_map['sentences']
            # batch_size = sentences.shape[0]
            length = sentences.shape[1]

            # Skip very short sentences.
            if length <= 2:
                continue

            result = trainer.step(batch_map, idx2word, train=False, compute_loss=True)
            recon_loss += result['reconstruction_softmax_loss']
            vg_loss += result.get('vg_loss', 0)
            contr_loss += result.get('contrastive_loss', 0)
            total_loss += result['total_loss']
            num_data += 1

            if diora.all_atten_score is not None:
                all_atten_score = torch.diagonal(diora.all_atten_score.cpu(), 0, 0, 1).permute(2, 0, 1)
            else:
                all_atten_score = None

            batch_ground_res = None
            if diora.atten_score is not None:
                targets = batch_map['VG_GT']
                batch_size = len(targets)
                attenion_scores = diora.atten_score.cpu()
                precomp_boxes = batch_map['boxes'].cpu()
                batch_ground_res = []
                for bid in range(batch_size):
                    target_bid, noun_mask = targets[bid]
                    precomp_boxes_bid = precomp_boxes[bid]
                    attenion_scores_bid = attenion_scores[bid]
                    # span_atten_scores = all_atten_score[bid]

                    ground_res = []
                    for _, gt_anno in target_bid.items():
                        start_id, end_id, gt_box = gt_anno
                        words_scores = attenion_scores_bid[start_id:end_id]
                        max_word_scores, _ = words_scores.max(1)
                        select_wid = max_word_scores.max(0)[1]
                        word2phr_atten = words_scores[select_wid]

                        # s, e = start_id, end_id-1
                        # k = e-s
                        # index = int(k*length - k*(k-1)/2 + s)
                        # span_atten = span_atten_scores[index]

                        select_box_ids = word2phr_atten.max(0)[1]
                        # select_box_ids = (word2phr_atten+0.1*span_atten).max(0)[1]
                        pred_box = precomp_boxes_bid[select_box_ids]

                        iou = torchops.box_iou(pred_box[None, :], torch.Tensor([gt_box]))
                        if iou.max() > 0.5:
                            recall_num += 1
                            ground_res.append(((start_id, end_id-1), 1))
                        else:
                            ground_res.append(((start_id, end_id-1), 0))
                        total_num += 1

                    batch_ground_res.append(ground_res)

            trees = parse_predictor.parse_batch(batch_map)
            for bid, tr in enumerate(trees):
                # CorpusF1
                gold_spans = set(batch_map['GT'][bid][:-1])
                pred_actions = get_actions(str(tr))
                pred_spans = set(get_spans(pred_actions)[:-1])
                tp, fp, fn = get_stats(pred_spans, gold_spans)
                corpus_f1[0] += tp
                corpus_f1[1] += fp
                corpus_f1[2] += fn

                # SentF1
                overlap = pred_spans.intersection(gold_spans)
                prec = float(len(overlap)) / (len(pred_spans) + 1e-8)
                reca = float(len(overlap)) / (len(gold_spans) + 1e-8)
                if len(gold_spans) == 0:
                    reca = 1.
                    if len(pred_spans) == 0:
                        prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)
                sent_f1.append(f1)

                # Ground Spans
                pred_boxes = []
                if all_atten_score is not None:
                    span_atten_scores = all_atten_score[bid]
                    word_atten_scores = attenion_scores[bid]
                    precomp_boxes_bid = precomp_boxes[bid]
                    for span in pred_spans:
                        s, e = span
                        k = e-s
                        index = int(k*length - k*(k-1)/2 + s)
                        span_atten = span_atten_scores[index]

                        word_atten = word_atten_scores[s:e+1]
                        max_word_scores, _ = word_atten.max(1)
                        select_wid = max_word_scores.max(0)[1]
                        word2span_atten = word_atten[select_wid]

                        select_box_ids = word2span_atten.max(0)[1]
                        # select_box_ids = (word2span_atten+0.1*span_atten).max(0)[1]
                        pred_box = precomp_boxes_bid[select_box_ids]
                        pred_boxes.append(pred_box.tolist())

                # CCRA
                if batch_ground_res is not None:
                    ground_res = batch_ground_res[bid]
                    for res in ground_res:
                        phr = res[0]
                        if res[1]:
                            if phr[1] == phr[0]:
                                ccr_num += 1
                            elif phr in pred_spans:
                                ccr_num += 1

                # write results
                example_id = batch_map['example_ids'][bid]
                s = [idx2word[idx] for idx in sentences[bid].tolist()]
                tr_index_conll = copy.deepcopy(tr)
                tr = replace_leaves(tr, s)
                if options.postprocess:
                    tr = postprocess(tr, s)
                # o = collections.OrderedDict(example_id=str(example_id), tree=tr)
                o = collections.OrderedDict(example_id=str(example_id), tree=tr, tree_index_conll=tr_index_conll,
                                            sentence=s, gold_spans=list(gold_spans), pred_spans=list(pred_spans),
                                            pred_boxes=pred_boxes)
                f.write(json.dumps(o) + '\n')

    f.close()
    ground_acc = recall_num / (total_num + 1e-8)
    ccra = ccr_num / (total_num + 1e-8)
    # print('grounding acc:{}'.format(ground_acc))
    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    print('corpus_f1:{} \t sent_f1:{} \t grounding acc:{} \t ccra:{}'.format(corpus_f1, sent_f1, ground_acc, ccra))
    print('recon_loss: {} ; vg_loss: {}; contr_loss: {}; total_loss: {}'.format(
        recon_loss/num_data, vg_loss/num_data, contr_loss/num_data, total_loss/num_data))


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
