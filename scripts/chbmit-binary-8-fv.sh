 uv run train.py --engine exp \
                --device mps \
                --num_epochs 200 \
                --batch_size 16 \
                --weight_decay 0 \
                --lr 0.001 \
                --experiment chbmit-binary-8 \
                --seq_len 1024 \
                --num_features 23 \
                --diff_order 5 \
                --k 0.9 \
                --t 40 \
                --d 4 \
                --dropout 0.9 \
                --d_model 2048 \
                --channel_independent False \
                --num_classes 2 \
                --task 2 \
                --dataset chbmit \
                --tolerance 75 \
                --recut True \
                --freq 256 \
                --is_three Fasle \
                --ts 20 \
                --tag train \
                --length 4 \
                --loss fl \
                --gamma 2 \
                --alpha 0.5 \
                --epsilon 0.05 \
                --log_step 20 \
                --root_path /Volumes/ersonlzy/datasets/chbmit \
                --model esdv1 \
                --resume True \
                --checkpoint runs/chbmit-binary-8-2025-06-08-22-37-04/checkpoints/checkpoint-160 \


                
