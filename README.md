

Launch the training with 4 GPUs
```
rm -rf logs
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed train.py --train_batch_size 256 \
    --train_micro_batch_size_per_gpu 64 --steps 200 \
    --run_name default --checkpoint_interval 100 
```
Resume the trainig from step 100 with 4 GPUs using ZeRO checkpoint
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed train.py --resume \
    --checkpoint_dir ds_transformer_checkpoint \
    --checkpoint_tag global_step100 \
    --train_batch_size 256 \
    --train_micro_batch_size_per_gpu 64 \
    --steps 200 \
    --checkpoint_interval 400 \
    --run_name resume_no_uc
```

Convert the zero checkpoint at step 100 to universal checkpoint
```
python ds_to_universal.py \
    --input_folder ds_transformer_checkpoint/global_step100/ \
    --output_folder ds_transformer_checkpoint/global_step100_universal \
    --num_extract_workers 1 --num_merge_workers 4  
```

Resume training with 4 GPUs with universal checkpoint
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed train.py --resume \
    --checkpoint_dir ds_transformer_checkpoint \
    --checkpoint_tag global_step100_universal \
    --train_batch_size 256 \
    --train_micro_batch_size_per_gpu 64 \
    --universal_checkpoint \
    --run_name resume_uc_4gpu \
    --steps 200 \
    --checkpoint_interval 400 
```

Resume training with 2 GPUs with gradient accumulation
```
CUDA_VISIBLE_DEVICES=0,1 deepspeed train.py --resume \
    --checkpoint_dir ds_transformer_checkpoint \
    --checkpoint_tag global_step100_universal \
    --train_batch_size 256 \
    --train_micro_batch_size_per_gpu 64 \
    --universal_checkpoint \
    --run_name resume_uc \
    --steps 200 \
    --checkpoint_interval 400 
```

