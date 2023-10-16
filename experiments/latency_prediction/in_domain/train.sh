BASE_DIR="/home/disk/NAR-Former-V2-github"
DATASET_DIR="$BASE_DIR/dataset/unseen_structure"

mkdir log
mkdir checkpoints

python $BASE_DIR/predictor/main.py \
    --gpu 2 \
    --lr 0.001 \
    --epochs 50 \
    --batch_size 16 \
    --data_root "$DATASET_DIR/data" \
    --all_latency_file "${DATASET_DIR}/gt_stage.txt" \
    --norm_sf \
    --onnx_dir "${DATASET_DIR}" \
    --log "log/train.log" \
    --model_dir "checkpoints/" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    --embed_type trans \
    --num_node_features 152 \
    --glt_norm LN \
    --warmup_rate 0.1 \
    --train_test_stage \
    --use_degree \