export CUDA_VISIBLE_DEVICES=0
python lm_finetuning.py \
--model_name_or_path=../model/RoBERTa_zh_Large_PyTorch \
--output_dir=../model/roberta_fine_8 \
--model_type=bert \
--do_train \
--train_data_file=../data/corpus.txt \
--eval_data_file=../data/corpus_test.txt \
--mlm \
--block_size=128 \
--per_gpu_train_batch_size=8 \
--num_train_epochs=2 \
--save_steps=2000 \
--weight_decay=1e-3 \
--learning_rate=5e-5 \
--warmup_proportion=0.1 \
--overwrite_output_dir \
--overwrite_cache;
