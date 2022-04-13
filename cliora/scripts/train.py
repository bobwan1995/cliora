import argparse

import os
import random
import uuid

import torch
import torchvision.ops as torchops
import numpy as np
from tqdm import tqdm
import sys
from cliora.data.dataset import ConsolidateDatasets, ReconstructDataset, make_batch_iterator

from cliora.utils.path import package_path
from cliora.logging.configuration import configure_experiment, get_logger
from cliora.utils.flags import stringify_flags, init_with_flags_file, save_flags
from cliora.utils.checkpoint import save_experiment

from cliora.net.experiment_logger import ExperimentLogger
from cliora.analysis.cky import ParsePredictor as CKY
from cliora.analysis.diora_tree import TreesFromDiora
from cliora.analysis.utils import *

data_types_choices = ('coco', 'flickr')


def count_params(net):
    return sum([x.numel() for x in net.parameters() if x.requires_grad])


def build_net(options, embeddings):
    from cliora.net.trainer import build_net

    trainer = build_net(options, embeddings, random_seed=options.seed)

    logger = get_logger()
    logger.info('# of params = {}'.format(count_params(trainer.net)))

    return trainer


def generate_seeds(n, seed=11):
    random.seed(seed)
    seeds = [random.randint(0, 2**16) for _ in range(n)]
    return seeds


def run_train(options, train_iterator, trainer, validation_iterator):
    logger = get_logger()
    experiment_logger = ExperimentLogger()
    save_emb = options.emb == 'none'

    logger.info('Running train.')

    seeds = generate_seeds(options.max_epoch, options.seed)
    word2idx = train_iterator.word2idx
    idx2word = {v: k for k, v in word2idx.items()}

    step = 0
    best_f1 = 0.
    # run_eval(options, trainer, validation_iterator)
    if not options.multigpu or options.local_rank == 0:
        if options.arch == 'hard':
            run_eval(options, trainer, validation_iterator)

    for epoch, seed in zip(range(options.max_epoch), seeds):
        # --- Train--- #

        # seed = seeds[epoch]

        logger.info('epoch={} seed={}'.format(epoch, seed))

        def myiterator():
            it = train_iterator.get_iterator(random_seed=seed)

            count = 0

            for batch_map in it:
                # TODO: Skip short examples (optionally).
                if batch_map['length'] <= 2:
                    continue

                yield count, batch_map
                count += 1

        for batch_idx, batch_map in myiterator():
            # if options.finetune and step >= options.finetune_after:
            #     trainer.freeze_diora()

            # trainer.py 359
            result = trainer.step(batch_map, idx2word)

            experiment_logger.record(result)

            if step % options.log_every_batch == 0:
                experiment_logger.log_batch(epoch, step, batch_idx, batch_size=options.batch_size)

            del result

            step += 1

        experiment_logger.log_epoch(epoch, step)

        # Epoch Eval and Checkpoints -- #
        if not options.multigpu or options.local_rank == 0:
            trainer.save_model(save_emb, os.path.join(options.experiment_path, 'model.epoch_{}.pt'.format(epoch)))
            save_experiment(os.path.join(options.experiment_path, 'experiment.epoch_{}.json'.format(epoch)), step)

            corpus_f1 = run_eval(options, trainer, validation_iterator)
            if corpus_f1 > best_f1:
                best_f1 = corpus_f1
            logger.info('Saving model epoch {},  corpus_f1: {}, best_f1: {}.'.format(epoch, corpus_f1, best_f1))

    if options.max_step is not None and step >= options.max_step:
            logger.info('Max-Step={} Quitting.'.format(options.max_step))
            sys.exit()


def run_eval(options, trainer, validation_iterator):
    logger = get_logger()

    # Eval mode.
    trainer.net.eval()
    if options.multigpu:
        diora = trainer.net.module.diora
    else:
        diora = trainer.net.diora
    # cliora.outside = False
    # cliora.outside = True # TODO
    diora.outside = options.obj_feats

    if options.arch == 'hard':
        diora.safe_set_K(2)
        parse_predictor = TreesFromDiora(net=diora)
    else:
        override_init_with_batch(diora)
        override_inside_hook(diora)
        parse_predictor = CKY(net=diora)
    batches = validation_iterator.get_iterator(random_seed=options.seed)

    logger.info('####### Beginning Eval #######')

    total_num = 0
    recall_num = 0
    corpus_f1 = [0., 0., 0.]
    sent_f1 = []
    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            sentences = batch_map['sentences']
            length = sentences.shape[1]

            # Skip very short sentences.
            if length <= 2:
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            # Grounding eval
            if diora.atten_score is not None:
                targets = batch_map['VG_GT']
                batch_size = len(targets)
                attenion_scores = diora.atten_score.cpu()
                precomp_boxes = batch_map['boxes'].cpu()
                for bid in range(batch_size):
                    target_bid, noun_mask = targets[bid]
                    precomp_boxes_bid = precomp_boxes[bid]
                    attenion_scores_bid = attenion_scores[bid]
                    select_scores, select_box_ids = attenion_scores_bid.max(1)
                    pred_boxes = precomp_boxes_bid[select_box_ids]

                    for _, gt_anno in target_bid.items():
                        start_id, end_id, gt_box = gt_anno
                        pred_box = pred_boxes[start_id:end_id]
                        select_score = select_scores[start_id:end_id]
                        select_id = select_score.max(0)[1]
                        iou = torchops.box_iou(pred_box[select_id][None, :], torch.Tensor([gt_box]))
                        if iou.max() > 0.5:
                            recall_num += 1
                        total_num += 1

            # Parsing eval
            trees = parse_predictor.parse_batch(batch_map)

            for bid, tr in enumerate(trees):
                # CorpusF1
                # gold_spans = set(batch_map['GT'][bid][5][:-1])
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

    ground_acc = recall_num / (total_num + 1e-8)
    # logger.info('grounding acc:{}'.format(ground_acc))
    # return ground_acc
    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    logger.info('corpus_f1:{} \t sent_f1:{} \t grounding acc:{}'.format(corpus_f1, sent_f1, ground_acc))

    # Train mode.
    diora.outside = True
    trainer.net.train()
    return corpus_f1


def get_train_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.train_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type=options.train_data_type)


def get_train_iterator(options, dataset):
    return make_batch_iterator(options, dataset, mode='train', shuffle=True,
            include_partial=False, filter_length=options.train_filter_length,
            batch_size=options.batch_size, length_to_size=options.length_to_size)


def get_validation_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.validation_path,
            embeddings_path=options.embeddings_path, filter_length=options.validation_filter_length,
            data_type=options.validation_data_type)


def get_validation_iterator(options, dataset):
    return make_batch_iterator(options, dataset, mode='test', shuffle=False,
            include_partial=True, filter_length=options.validation_filter_length,
            batch_size=options.validation_batch_size, length_to_size=options.length_to_size)


def get_train_and_validation(options):
    train_dataset = get_train_dataset(options)
    validation_dataset = get_validation_dataset(options)

    # Modifies datasets. Unifying word mappings, embeddings, etc.
    if options.data_type not in ['coco', 'flickr']:
        ConsolidateDatasets([train_dataset, validation_dataset]).run()

    return train_dataset, validation_dataset


def run(options):
    logger = get_logger()
    # experiment_logger = ExperimentLogger()

    train_dataset, validation_dataset = get_train_and_validation(options)
    if options.debug:
        train_iterator = get_validation_iterator(options, validation_dataset)
    else:
        train_iterator = get_train_iterator(options, train_dataset)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    embeddings = train_dataset['embeddings']

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings)
    logger.info('Model:')
    for name, p in trainer.net.named_parameters():
        logger.info('{} {} {}'.format(name, p.shape, p.requires_grad))

    run_train(options, train_iterator, trainer, validation_iterator)


def argument_parser():
    parser = argparse.ArgumentParser()

    # Debug.
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--git_sha', default=None, type=str)
    parser.add_argument('--git_branch_name', default=None, type=str)
    parser.add_argument('--git_dirty', default=None, type=str)
    parser.add_argument('--uuid', default=None, type=str)
    parser.add_argument('--model_flags', default=None, type=str,
                        help='Load model settings from a flags file.')
    parser.add_argument('--flags', default=None, type=str,
                        help='Load any settings from a flags file.')

    parser.add_argument('--master_addr', default='127.0.0.1', type=str)
    parser.add_argument('--master_port', default='29500', type=str)
    parser.add_argument('--world_size', default=None, type=int)

    # Pytorch
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument("--local_rank", default=None, type=int) # for distributed-data-parallel

    # Logging.
    parser.add_argument('--default_experiment_directory', default=os.path.join(package_path(), '..', 'log'), type=str)
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--experiment_path', default=None, type=str)
    parser.add_argument('--log_every_batch', default=10, type=int)
    parser.add_argument('--save_latest', default=1000, type=int)
    parser.add_argument('--save_distinct', default=5000, type=int)
    parser.add_argument('--save_after', default=1000, type=int)

    # Loading.
    parser.add_argument('--load_model_path', default=None, type=str)

    # Data.
    parser.add_argument('--data_type', default='nli', choices=data_types_choices)
    parser.add_argument('--train_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--validation_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--train_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_train.jsonl'), type=str)
    parser.add_argument('--validation_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--embeddings_path', default=os.path.expanduser('~/data/glove/glove.6B.300d.txt'), type=str)

    # Data (synthetic).
    parser.add_argument('--synthetic-nexamples', default=1000, type=int)
    parser.add_argument('--synthetic-vocabsize', default=1000, type=int)
    parser.add_argument('--synthetic-embeddingsize', default=1024, type=int)
    parser.add_argument('--synthetic-minlen', default=20, type=int)
    parser.add_argument('--synthetic-maxlen', default=21, type=int)
    parser.add_argument('--synthetic-seed', default=11, type=int)
    parser.add_argument('--synthetic-length', default=None, type=int)
    parser.add_argument('--use-synthetic-embeddings', action='store_true')

    # Data (preprocessing).
    parser.add_argument('--uppercase', action='store_true')
    parser.add_argument('--train_filter_length', default=50, type=int)
    parser.add_argument('--validation_filter_length', default=0, type=int)

    # Model.
    parser.add_argument('--arch', default='mlp', choices=('mlp', 'hard'))
    parser.add_argument('--share', action='store_false')
    parser.add_argument('--hidden_dim', default=400, type=int)
    parser.add_argument('--normalize', default='unit', choices=('none', 'unit'))
    parser.add_argument('--compress', action='store_true',
                        help='If true, then copy root from inside chart for outside. ' + \
                             'Otherwise, learn outside root as bias.')

    # Model (Objective).
    parser.add_argument('--reconstruct_mode', default='softmax',
                        choices=('softmax'))

    # Model (Embeddings).
    parser.add_argument('--emb', default='w2v', choices=('w2v', 'skip', 'elmo', 'both', 'none'))

    # Model (Negative Sampler).
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--k_neg', default=100, type=int)
    parser.add_argument('--freq_dist_power', default=0.75, type=float)

    # ELMo
    parser.add_argument('--elmo_options_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weights_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    parser.add_argument('--elmo_cache_dir', default='./log/elmo', type=str,
                        help='If set, then context-insensitive word embeddings will be cached ' + \
                             '(identified by a hash of the vocabulary).')

    # Training.
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--length_to_size', default=None, type=str,
                        help='Easily specify a mapping of length to batch_size.' + \
                             'For instance, 10:32,20:16 means that all batches' + \
                             'of length 10-19 will have batch size 32, 20 or greater' + \
                             'will have batch size 16, and less than 10 will have batch size' + \
                             'equal to the batch_size arg. Only applies to training.')
    parser.add_argument('--train_dataset_size', default=None, type=int)
    parser.add_argument('--validation_dataset_size', default=None, type=int)
    parser.add_argument('--validation_batch_size', default=None, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--max_step', default=None, type=int)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_after', default=0, type=int)

    # Parsing.
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    # Optimization.
    parser.add_argument('--lr', default=2e-3, type=float)

    # Vis feature
    parser.add_argument('--alpha_contr', type=float, default=1.0)
    parser.add_argument('--obj_feats', action='store_true')
    parser.add_argument('--vl_margin', default=0.2, type=float)
    parser.add_argument('--use_contr', action='store_true')
    parser.add_argument('--use_contr_ce', action='store_true')
    parser.add_argument('--vg_loss', action='store_true')
    parser.add_argument('--alpha_vg', type=float, default=1.0)
    parser.add_argument('--alpha_kl', type=float, default=1.0)

    # S-DIORA
    parser.add_argument('--hinge_margin', default=1, type=float)

    return parser


def parse_args(parser):
    options, other_args = parser.parse_known_args()

    # Set default flag values (data).
    options.train_data_type = options.data_type if options.train_data_type is None else options.train_data_type
    options.validation_data_type = options.data_type if options.validation_data_type is None else options.validation_data_type
    options.validation_batch_size = options.batch_size if options.validation_batch_size is None else options.validation_batch_size

    # Set default flag values (config).
    if not options.git_branch_name:
        options.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not options.git_sha:
        options.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not options.git_dirty:
        options.git_dirty = os.popen("git diff --quiet && echo 'clean' || echo 'dirty'").read().strip()

    if not options.uuid:
        options.uuid = str(uuid.uuid4())

    if not options.experiment_name:
        options.experiment_name = '{}'.format(options.uuid[:8])

    if not options.experiment_path:
        options.experiment_path = os.path.join(options.default_experiment_directory, options.experiment_name)

    if options.length_to_size is not None:
        parts = [x.split(':') for x in options.length_to_size.split(',')]
        options.length_to_size = {int(x[0]): int(x[1]) for x in parts}

    options.lowercase = not options.uppercase

    for k, v in options.__dict__.items():
        if type(v) == str and v.startswith('~'):
            options.__dict__[k] = os.path.expanduser(v)

    # Load model settings from a flags file.
    if options.model_flags is not None:
        flags_to_use = []
        flags_to_use += ['arch']
        flags_to_use += ['compress']
        flags_to_use += ['emb']
        flags_to_use += ['hidden_dim']
        flags_to_use += ['normalize']
        flags_to_use += ['reconstruct_mode']

        options = init_with_flags_file(options, options.model_flags, flags_to_use)

    # Load any setting from a flags file.
    if options.flags is not None:
        options = init_with_flags_file(options, options.flags)

    return options


def configure(options):
    # Configure output paths for this experiment.
    configure_experiment(options.experiment_path, rank=options.local_rank)

    # Get logger.
    logger = get_logger()

    # Print flags.
    logger.info(stringify_flags(options))
    save_flags(options, options.experiment_path)


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
