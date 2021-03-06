## CLIORA

This is the official codebase for ICLR **oral** paper: Unsupervised Vision-Language Grammar Induction with Shared Structure Modeling. 

We introduce a new task of Unsupervised Vision-Language Grammar Induction and devise a model Contrastive Language-Image inside-Outside Recursive Autoencoder (CLIORA) to solve it. Please read our paper for more details: https://openreview.net/forum?id=N0n_QyQ5lBF.

This code follows the implementation architecture of [DIORA](https://github.com/iesl/diora).

Please cite our paper as follows:

```
@inproceedings{wan2022cliora,
  title={Unsupervised Vision-Language Grammar Induction with Shared Structure Modeling},
  author={Wan, Bo and Han, Wenjuan and Zheng, Zilong and Tuytelaars, Tinne},
  booktitle={The International Conference on Learning Representations (ICLR)},
  year={2022},
}
```

## Envs and Datas

Install dependencies (using Conda as a virtual environment):

```
conda create -n cliora python=3.8
source activate cliora
pip install -r requirements.txt
```


Download [flickr_data](https://esatkuleuvenbe-my.sharepoint.com/:u:/g/personal/bwan_esat_kuleuven_be/ERcLeIlPJxBDg7Jdf6IwOT0BU5kbcTHSRM7U_dPX_y4ftg?e=j9gyB9) and [outputs](https://esatkuleuvenbe-my.sharepoint.com/:u:/g/personal/bwan_esat_kuleuven_be/EYCdZiPIcj5OtQQqIH49B4gBcfT607sKdnGxrsdkYPKapQ?e=1aGlyk) and put the files as the following structure:

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

We use the same object features as [MAF](https://github.com/qinzzz/Multimodal-Alignment-Framework). Download [train_features_compress.hdf5](https://drive.google.com/file/d/1ABnF0SZMf6pOAC89LJXbXZLMW1X86O96/view?usp=sharing), [val features_compress.hdf5](https://drive.google.com/file/d/1iK-yz6PHwRuAciRW1vGkg9Bkj-aBE8yJ/view?usp=sharing), [test features_compress.hdf5](https://drive.google.com/file/d/1pjntkbr20l2MiUBVQLVV6rQNWpXQymFs/view?usp=sharing) to `flickr_data/flickr_feat_maf`.

## Running CLIORA
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
Single-GPU training:
```
export CUDA_VISIBLE_DEVICES=0
python -m cliora/scripts/train.py
    --cuda
    ... # other args
```

Multi-GPU Training:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS cliora/scripts/train.py
    --cuda
    --multigpu
    ... # other args
```

## Visualization
Download [Flickr30K Entities Dataset](http://hockenmaier.cs.illinois.edu/DenotationGraph/) and put the image folder `flickr_images` under `flickr_data/`. Add `--visualize` when run `test_cliora.sh`:
```
# test_cliora.sh
python cliora/scripts/parse.py
    --cuda
    --visualize
    --obj_feats
    ... # other args
```

## Word Embedding

We provide randomly-initialized word embedding, skip-thoughts embedding and ELMo embedding. If you use ELMo embedding and specify the `--elmo_cache_dir`, then the context-insensitive ELMo vectors will be cached, making it much faster to load these vectors after the initial usage.

Example Usage:

```
word_emb=none/skip/elmo

python cliora/scripts/train.py
    --emb word_emb
    ... # other args
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
