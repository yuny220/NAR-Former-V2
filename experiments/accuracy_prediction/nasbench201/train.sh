BASE_DIR="/home/disk/NAR-Former-V2-github"
DATASET_DIR="$BASE_DIR/dataset/nasbench201"

mkdir log
mkdir checkpoints

python $BASE_DIR/predictor/main.py \
    --gpu 1 \
    --lr 0.0001 \
    --warmup_rate 0.1 \
    --epochs 1000 \
    --batch_size 8 \
    --dataset "nasbench201" \
    --train_num 781 \
    --data_root "$DATASET_DIR/all.pt" \
    --log "log/train.log" \
    --model_dir "checkpoints/" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    --embed_type nerf \
    --num_node_features 128 \
    --n_attned_gnn 6 \
    --glt_norm LN \
    --use_degree \
    --multires_x 32 \
    --multires_p 32 \
    --optype pe \
    --lambda_sr 0.1 \
    --lambda_cons 0.5 \
    #--resume "checkpoints/$TEST_MODEL_TYPE/ckpt_best.pth" \
    #--only_test \
    #--override_data