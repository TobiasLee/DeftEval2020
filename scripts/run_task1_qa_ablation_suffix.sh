GPU="2,3,4,5"
TRAIN_DIR=data/train
DEV_DIR=data/dev
TEST_DIR=data/test
EPOCH=3

LR=2e-5
MODEL=roberta-base
BSZ=32


OUTPUT_DIR=results/bullshit-suffix-ablation-$MODEL-$LR-$BSZ
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
         --do_train --do_eval --do_predict --warmup_ratio 0.1 \
         --qa_type --question_position suffix