 uv run train.py --engine cla \
                --device cuda \
                --num_epochs 200 \
                --batch_size 64 \
                --weight_decay 0 \
                --lr 0.0001 \
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
                --is_three False \
                --ts 0 \
                --tag train \
                --length 4 \
                --loss ce \
                --gamma 3 \
                --alpha 0.5 \
                --epsilon 0.05 \
                --log_step 100 \
                --root_path /root/autodl-tmp/datasets/chbmit \


                
