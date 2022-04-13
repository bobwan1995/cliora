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
            vg_loss += result['vg_loss']
            contr_loss += result['contrastive_loss']
            total_loss += result['total_loss']
            num_data += 1

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
                                            )
                f.write(json.dumps(o) + '\n')

    f.close()
    # print('grounding acc:{}'.format(ground_acc))
    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    print('corpus_f1:{} \t sent_f1:{}'.format(corpus_f1, sent_f1))
    print('recon_loss: {} ; vg_loss: {}; contr_loss: {}; total_loss: {}'.format(
        recon_loss/num_data, vg_loss/num_data, contr_loss/num_data, total_loss/num_data))


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
