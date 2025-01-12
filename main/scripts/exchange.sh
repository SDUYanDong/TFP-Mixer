if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi


# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY=
export WANDB_MODE=offline

model_name=TFPMixer
pred_lens=(96 192 336 720)
cuda_ids1=(0 1 2 3)



for ((i = 0; i < 4; i++))
do
    pred_len=${pred_lens[i]}
    export CUDA_VISIBLE_DEVICES=${cuda_ids1[1]}
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id exchange_rate_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --e_layers 7 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 12 \
    --dropout 0.4\
    --fc_dropout 0.4\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100  \
    --itr 1 --batch_size 128 --learning_rate 0.00001 \
    --dp_rank 8 --warmup_epochs 0 \
    2>&1 | tee logs/LongForecasting/$model_name'_'exchange_rate_96_$pred_len.log &\

  done

