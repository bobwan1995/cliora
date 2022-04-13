#export PYTHONPATH=$(pwd):$PYTHONPATH
EXP_PATH="./outputs/flickr/flickr_diora_5e4_mlpshare_bs32_RandInit_seed1234"
export CUDA_VISIBLE_DEVICES=0
python cliora/scripts/parse.py \
    --cuda \
    --arch mlp \
    --batch_size 64 \
    --emb none \
    --embeddings_path none \
    --hidden_dim 400 \
    --k_neg 100 \
    --log_every_batch 100 \
    --normalize unit \
    --reconstruct_mode softmax \
    --data_type flickr \
    --train_path ./flickr_data/flickr_train.json \
    --validation_path ./flickr_data/flickr_test.json \
    --experiment_path  $EXP_PATH \
    --load_model_path $EXP_PATH/model.epoch_29.pt