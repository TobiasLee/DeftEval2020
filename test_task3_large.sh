GPU="2"

TRAIN_FILE=data/subtask3/train.tsv
DEV_FILE=data/subtask3/dev.tsv
TEST_FILE=data/subtask3/test.tsv

EPOCH=5

LR=2e-5
MODEL=roberta-large  #base
BSZ=4

MAX_LEN=256 
MODEL_DIR=ckpts/task3-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python3 task3.py --train_file $TRAIN_FILE \
         --dev_file $DEV_FILE \
         --test_file $TEST_FILE --fp16 --max_seq_length $MAX_LEN  \
         --overwrite_cache \
         --num_train_epochs $EPOCH \
         --output_dir $MODEL_DIR \
         --model_name_or_path tobiaslee/roberta-large-defteval-t6-st3  \
         --learning_rate $LR --evaluation_strategy steps \
         --metric_for_best_model "f1" \
         --load_best_model_at_end \
         --save_total_limit 2 \
         --logging_steps 100 \
         --per_device_train_batch_size $BSZ \
         --per_device_eval_batch_size $BSZ \
         --do_eval --do_predict --warmup_ratio 0.1
