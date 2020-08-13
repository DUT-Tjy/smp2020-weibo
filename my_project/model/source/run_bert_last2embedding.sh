export CUDA_VISIBLE_DEVICES=0
for i in `seq 0 4`;
do

python run_bert_2562_last2embedding_cls.py \
--model_type bert \
--model_name_or_path ../model/roberta_fine_8 \
--do_train \
--do_eval \
--do_test \
--data_dir ../data/data_StratifiedKFold_42/data_origin_$i \
--output_dir ../model/roberta_fine_last2embedding/bert_large_$i \
--max_seq_length 256 \
--split_num 1 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 40000 \
--not_do_eval_steps 0.5 \
--freeze 0

done

