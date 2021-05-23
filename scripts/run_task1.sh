GPU="1,2,3,4"
TRAIN_DIR=data/subtask1/train
DEV_DIR=data/subtask1/dev
TEST_DIR=data/subtask1/test
EPOCH=3

LR=2e-5
MODEL=roberta-large
BSZ=32


OUTPUT_DIR=results/subtask1-$MODEL-$LR-$BSZ
CUDA_VISIBLE_DEVICES=$GPU python3 task1.py --train_dir $TRAIN_DIR \
         --dev_dir $DEV_DIR \
         --test_dir $TEST_DIR \
         --overwrite_cache \
         --num_train_epochs $EPOCH \
         --output_dir $OUTPUT_DIR \
         --model_name_or_path $MODEL  \
         --learning_rate $LR --evaluation_strategy steps \
         --metric_for_best_model "f1" \
         --load_best_model_at_end \
         --save_total_limit 2 \
         --logging_steps 5 \
         --per_device_train_batch_size $BSZ \
         --per_device_eval_batch_size $BSZ \
         --do_train --do_eval --do_predict --warmup_ratio 0.1

# evaluate on test set only
#MODEL=results/roberta-base-2e-5-32/
#CUDA_VISIBLE_DEVICES=$GPU python3 task1.py --train_dir $TRAIN_DIR \
#         --dev_dir $DEV_DIR \
#         --test_dir $TEST_DIR \
#         --overwrite_cache \
#         --num_train_epochs $EPOCH \
#         --output_dir $OUTPUT_DIR \
#         --model_name_or_path $MODEL  \
#         --learning_rate $LR --evaluation_strategy steps \
#         --metric_for_best_model "f1" \
#         --load_best_model_at_end \
#         --save_total_limit 2 \
#         --logging_steps 5 \
#         --per_device_train_batch_size $BSZ \
#         --per_device_eval_batch_size $BSZ \
#         --do_predict --warmup_ratio 0.1