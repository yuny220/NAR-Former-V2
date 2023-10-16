BASE_DIR="/home/disk/NAR-Former-V2-github"
DATASET_DIR="$BASE_DIR/dataset/nasbench201"

python $BASE_DIR/predictor/main.py \
    --only_test \
    --gpu 1 \
    --lr 0.0001 \
    --warmup_rate 0.1 \
    --epochs 1000 \
    --batch_size 1024 \
    --dataset "nasbench201" \
    --train_num 781 \
    --data_root "$DATASET_DIR/all.pt" \
    --log "log/test.log" \
    --pretrain "checkpoints/ckpt_best.pth" \
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