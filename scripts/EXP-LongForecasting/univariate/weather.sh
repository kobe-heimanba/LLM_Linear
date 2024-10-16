
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=LLM_Linear

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 128 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'96'_'100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --AutoCon_wnorm ReVIN \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 64 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'192'_'100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 336 \
  --AutoCon_wnorm Mean \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 64 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'336'_'100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 1 \
  --c_out 1 \
  --AutoCon_wnorm Mean \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 64 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'720'_'100.log

python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/weather/ \
 --data_path weather.csv \
 --model_id weather_$seq_len'_'96 \
 --model $model_name \
 --data custom \
 --features S \
 --seq_len $seq_len \
 --pred_len 96 \
 --enc_in 1 \
 --des 'Exp' \
 --percent 10 \
 --AutoCon_wnorm Mean \
 --c_out 1 \
 --itr 1 --batch_size 8 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'96'_'10.log

python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/weather/ \
 --data_path weather.csv \
 --model_id weather_$seq_len'_'192 \
 --model $model_name \
 --data custom \
 --features S \
 --seq_len $seq_len \
 --pred_len 192 \
 --enc_in 1 \
 --des 'Exp' \
 --percent 10 \
 --AutoCon_wnorm ReVIN \
 --c_out 1 \
 --itr 1 --batch_size 8 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'192'_'10.log

python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/weather/ \
 --data_path weather.csv \
 --model_id weather_$seq_len'_'336 \
 --model $model_name \
 --data custom \
 --features S \
 --seq_len $seq_len \
 --pred_len 336 \
 --enc_in 1 \
 --des 'Exp' \
 --percent 10 \
 --AutoCon_wnorm Mean \
 --c_out 1 \
 --itr 1 --batch_size 8 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'336'_'10.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --AutoCon_wnorm Mean \
  --c_out 1 \
  --itr 1 --batch_size 64 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'720'_'10.log

