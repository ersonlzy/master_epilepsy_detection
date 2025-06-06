 uv run train.py --engine cla \
                --device mps \
                --num_epochs 200 \
                --batch_size 64 \
                --weight_decay 0.0 \
                --lr 0.001 \
                --experiment bonn-binary-abcd-e-8 \
                --seq_len 512 \
                --num_features 1 \
                --diff_order 5 \
                --k 0.9 \
                --t 25 \
                --d 5 \
                --dropout 0.5 \
                --d_model 1024 \
                --channel_independent True \
                --num_classes 2 \
                --task 2 \
                --dataset bonn \
                --tolerance 75 \
                --recut True \
                --freq 173.6 \
                --is_three True \
                --ts 20 \
                --tag train \
                --length 4 \
                --loss fl \
                --gamma 2 \
                --alpha 0.5 \
                --epsilon 0.05 \
                --log_step 20 \
                --root_path /Volumes/ersonlzy/datasets/bonn \
                --datasets_task 0 \
                --model esdv1


                
