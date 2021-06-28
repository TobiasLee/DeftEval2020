GPU="0"
TRAIN_DIR=data/ori_data/train
DEV_DIR=data/ori_data/dev
TEST_DIR=data/subtask2/test
EPOCH=10

LR=3e-5
MODEL=roberta-large
BSZ=32


MODEL_DIR=ckpts/task2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python3 task2.py --train_dir $TRAIN_DIR \
         --dev_dir $DEV_DIR \
         --test_dir $TEST_DIR \
         --overwrite_cache \
         --num_train_epochs $EPOCH \
         --output_dir $MODEL_DIR \
         --model_name_or_path tobiaslee/roberta-large-defteval-t6-st2  \
         --learning_rate $LR --evaluation_strategy steps \
         --metric_for_best_model "f1" \
         --load_best_model_at_end \
         --save_total_limit 2 \
         --logging_steps 50 \
         --per_device_train_batch_size $BSZ \
         --per_device_eval_batch_size $BSZ \
         --do_predict --warmup_ratio 0.1
