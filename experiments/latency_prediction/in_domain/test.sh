BASE_DIR="/home/disk/NAR-Former-V2-github"
DATASET_DIR="$BASE_DIR/dataset/unseen_structure"

python $BASE_DIR/predictor/main.py \
    --only_test \
    --gpu 3 \
    --batch_size 1024 \
    --data_root "$DATASET_DIR/data" \
    --all_latency_file "${DATASET_DIR}/gt_stage.txt" \
    --norm_sf \
    --onnx_dir "${DATASET_DIR}" \
    --log "log/test.log" \
    --pretrain "checkpoints/ckpt_best.pth" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    --embed_type trans \
    --num_node_features 152 \
    --glt_norm LN \
    --train_test_stage \
    --use_degree \