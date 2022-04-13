# *****************  CLIORA  *******************

# Finetune CLIORA based on DIORA on Flickr30K with word embedding from pretrained DIORA
EXP_PATH="./outputs/flickr/flickr_cliora_1e5_mlpshare_bs32_RandInit_seed1234_valid"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 cliora/scripts/train.py \
    --cuda --multigpu \
    --max_epoch 10 \
    --seed 1234 \
    --arch mlp \
    --batch_size 32 \
    --emb none \
    --embeddings_path none \
    --hidden_dim 400 \
    --k_neg 100 \
    --log_every_batch 100 \
    --lr 1e-5 \
    --normalize unit \
    --reconstruct_mode softmax \
    --train_filter_length 40 \
    --data_type flickr \
    --train_path ./flickr_data/flickr_train.json \
    --validation_path ./flickr_data/flickr_test.json \
    --experiment_path  $EXP_PATH \
    --obj_feats \
    --use_contr --alpha_contr 1.0 \
    --vg_loss --alpha_vg 1.0 \
    --load_model_path ./outputs/flickr/flickr_diora_5e4_mlpshare_bs32_RandInit_seed1234/model.epoch_29.pt


# Finetune CLIORA based on DIORA on Flickr30K with skip-thoughts initialized word embedding
EXP_PATH="./outputs/flickr/flickr_cliora_1e5_mlpshare_bs32_skip_seed1234"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 diora/scripts/train.py \
    --cuda --multigpu \
    --max_epoch 10 \
    --seed 1234 \
    --master_port 12345 \
    --arch mlp \
    --batch_size 32 \
    --emb skip \
    --embeddings_path ./flickr_data/skip_thoughts_dict.pkl \
    --hidden_dim 400 \
    --k_neg 100 \
    --log_every_batch 100 \
    --lr 1e-5 \
    --normalize unit \
    --reconstruct_mode softmax \
    --train_filter_length 40 \
    --data_type flickr \
    --train_path ./flickr_data/flickr_train.json \
    --validation_path ./flickr_data/flickr_test.json \
    --experiment_path  $EXP_PATH \
    --obj_feats \
    --use_contr --alpha_contr 1.0 \
    --vg_loss --alpha_vg 1.0 \
    --load_model_path ./outputs/flickr/flickr_diora_5e4_mlpshare_bs64_skip_seed1234/model.epoch_29.pt
