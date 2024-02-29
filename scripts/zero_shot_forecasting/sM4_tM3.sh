export CUDA_VISIBLE_DEVICES=0

model_name=AutoTimes_Llama

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --test_data_path m3_yearly_dataset.tsf \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data tsf \
  --seq_len 12 \
  --label_len 6 \
  --token_len 6 \
  --test_seq_len 12 \
  --test_label_len 6 \
  --test_pred_len 6 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --test_dir short_term_forecast_m4_Yearly_AutoTimes_Llama_m4_sl12_ll6_tl6_lr0.0001_bt16_wd1e-05_hd512_hl2_cosTrue_mixFalse_Exp_0

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --test_data_path m3_quarterly_dataset.tsf \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data tsf \
  --seq_len 16 \
  --label_len 8 \
  --token_len 8 \
  --test_seq_len 16 \
  --test_label_len 8 \
  --test_pred_len 8 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --test_dir short_term_forecast_m4_Quarterly_AutoTimes_Llama_m4_sl16_ll8_tl8_lr5e-05_bt16_wd5e-06_hd512_hl2_cosTrue_mixFalse_Exp_0

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --test_data_path m3_monthly_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data tsf \
  --seq_len 36 \
  --label_len 18 \
  --token_len 18 \
  --test_seq_len 36 \
  --test_label_len 18 \
  --test_pred_len 18 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --test_dir short_term_forecast_m4_Monthly_AutoTimes_Llama_m4_sl36_ll18_tl18_lr5e-05_bt16_wd1e-06_hd1024_hl2_cosTrue_mixFalse_Exp_0

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --test_data_path m3_other_dataset.tsf \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data tsf \
  --seq_len 16 \
  --label_len 8 \
  --token_len 8 \
  --test_seq_len 16 \
  --test_label_len 8 \
  --test_pred_len 8 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --test_dir short_term_forecast_m4_Quarterly_AutoTimes_Llama_m4_sl16_ll8_tl8_lr5e-05_bt16_wd5e-06_hd512_hl2_cosTrue_mixFalse_Exp_0