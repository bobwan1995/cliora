## CLIORA

This is the official codebase for ICLR **oral** paper Unsupervised Vision-Language Grammar Induction with Shared Structure Modeling. We introduce a new task of Unsupervised Vision-Language Grammar Induction and devise a model Contrastive Language-Image inside-Outside Recursive Autoencoder (CLIORA) to solve it. Please read our paper for more details: https://openreview.net/forum?id=N0n_QyQ5lBF.

This code follows the implementation architecture of [DIORA](https://github.com/iesl/diora).

## Quick Start

Install dependencies (using Conda as a virtual environment).

```
conda create -n cliora python=3.8
source activate cliora
pip install -r requirements.txt
```


Download the [datasets]() and the [checkpoints](https://esatkuleuvenbe-my.sharepoint.com/:u:/g/personal/bwan_esat_kuleuven_be/EYCdZiPIcj5OtQQqIH49B4gBcfT607sKdnGxrsdkYPKapQ?e=1aGlyk).



Make sure to put the files as the following structure:

```
  cliora
  ├───cliora
  │   ├─...
  │
  ├───flickr_data
  │   ├─flickr_feat_maf
  │
  ├───outputs
      ├─flickr
```

Running CLIORA.
```
export PYTHONPATH=$(pwd):$PYTHONPATH


## Train DIORA
sh train_diora.sh

## Test DIORA
sh test_diora.sh

## Train CLOIRA based on DIORA
sh train_clora.sh

## Test CLIORA 
sh test_cliora.sh
```

## Multi-GPU Training

Using `DistributedDataParallel`:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS diora/scripts/train.py \
    --cuda \
    --multigpu \
    ... # other args
```


## Word Embedding

We provide randomly-initialized word embedding, skip-thoughts embedding and elmo embedding. If you use elmo embedding and specify the `--elmo_cache_dir`, then the context-insensitive ELMo vectors will be cached, making it much faster to load these vectors after the initial usage.

Example Usage:

```
word_emb=none/skip/elmo

python cliora/scripts/train.py \
    --emb $word_emb \
    ... # other args
```



Please cite our paper as follows:

```
@inproceedings{wan2022cliora,
  title={Unsupervised Vision-Language Grammar Induction with Shared Structure Modeling},
  author={Wan, Bo and Han, Wenjuan and Zheng, Zilong and Tuytelaars, Tinne},
  booktitle={The International Conference on Learning Representations (ICLR)},
  year={2022},
}
```

## License

Copyright 2018, University of Massachusetts Amherst

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
