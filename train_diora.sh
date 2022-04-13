# *****************  DIORA  *******************
# Train original DIORA on Flickr30K with randomly-initialized word embedding
# For randomly-initialized word embedding, bs 32 gets better results than bs64
EXP_PATH="./outputs/flickr/flickr_diora_5e4_mlpshare_bs32_RandInit_seed1234"

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 diora/scripts/train.py \
    --cuda --multigpu \
    --max_epoch 30 \
    --seed 1234 \
    --arch mlp \
    --batch_size 32 \
    --emb none \
    --embeddings_path none \
    --hidden_dim 400 \
    --k_neg 100 \
    --log_every_batch 100 \
    --lr 5e-4 \
    --normalize unit \
    --reconstruct_mode softmax \
    --train_filter_length 40 \
    --data_type flickr \
    --train_path ./flickr_data/flickr_train.json \
    --validation_path ./flickr_data/flickr_test.json \
    --experiment_path  $EXP_PATH


# Train original DIORA on Flickr30K with skip-thoughts initialized word embedding
EXP_PATH="./outputs/flickr/flickr_diora_5e4_mlpshare_bs64_skip_seed1234"

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 cliora/scripts/train.py \
    --cuda --multigpu \
    --max_epoch 30 \
    --seed 1234 \
    --arch mlp \
    --batch_size 64 \
    --emb skip \
    --embeddings_path ./flickr_data/skip_thoughts_dict.pkl \
    --hidden_dim 400 \
    --k_neg 100 \
    --log_every_batch 100 \
    --lr 5e-4 \
    --normalize unit \
    --reconstruct_mode softmax \
    --train_filter_length 40 \
    --data_type flickr \
    --train_path ./flickr_data/flickr_train.json \
    --validation_path ./flickr_data/flickr_test.json \
    --experiment_path  $EXP_PATH