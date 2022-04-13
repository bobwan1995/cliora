
import torch
import numpy as np
from tqdm import tqdm

from train import argument_parser, parse_args, configure
from train import get_validation_dataset, get_validation_iterator

from cliora.analysis.utils import *

def run(options):
    validation_dataset = get_validation_dataset(options)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    batches = validation_iterator.get_iterator(random_seed=options.seed)
    print('Beginning.')
    corpus_f1 = [0., 0., 0.]
    sent_f1 = []

    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            sentences = batch_map['sentences']
            batch_size = sentences.shape[0]
            length = sentences.shape[1]

            # Skip very short sentences.
            if length < 2:
                continue

            for bid in range(batch_size):
                # CorpusF1
                # gold_spans = set(batch_map['GT'][bid][5][:-1])
                gold_spans = set(batch_map['GT'][bid][:-1])
                # right branch
                pred_span = [(i, length-1) for i in range(length-1)]
                pred_spans = set(pred_span[1:])
                # left branch
                # pred_spans = set([(0, i) for i in range(1, length-1)])
                # tp, fp, fn = get_stats(pred_spans, gold_spans)
                tp = len(gold_spans); fp = len(pred_spans) - tp; fn = 0
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


    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    print('corpus_f1:{} \t sent_f1:{}'.format(corpus_f1, sent_f1))


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
