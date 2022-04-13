#export PYTHONPATH=$(pwd):$PYTHONPATH
EXP_PATH="./outputs/flickr/flickr_cliora_1e5_mlpshare_bs32_skip_seed1234"
#python cliora/scripts/right_branch.py \
#for ckpt in "0.pt" "1.pt" "2.pt" "3.pt" "4.pt"
#do
ckpt="1.pt"
export CUDA_VISIBLE_DEVICES=0
python cliora/scripts/parse.py \
    --cuda \
    --arch mlp \
    --batch_size 64 \
    --emb skip \
    --embeddings_path ./flickr_data/skip_thoughts_dict.pkl \
    --hidden_dim 400 \
    --k_neg 100 \
    --log_every_batch 100 \
    --normalize unit \
    --reconstruct_mode softmax \
    --data_type flickr \
    --train_path ./flickr_data/flickr_train.json \
    --validation_path ./flickr_data/flickr_test.json \
    --experiment_path  $EXP_PATH \
    --obj_feats \
    --use_contr \
    --vg_loss \
    --load_model_path $EXP_PATH/model.epoch_$ckpt
#done