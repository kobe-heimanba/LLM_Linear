
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
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_$seq_len'_'96 \
 --model $model_name \
 --data traffic \
 --features MS \
 --seq_len $seq_len \
 --pred_len 96 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 1 \
 --des 'Exp' \
 --percent 100 \
 --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'96'_'100.log

python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_$seq_len'_'192 \
 --model $model_name \
 --data traffic \
 --features MS \
 --seq_len $seq_len \
 --pred_len 192 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 1 \
 --des 'Exp' \
 --percent 100 \
 --itr 1 --batch_size 16    >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'192'_'100.log

python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_$seq_len'_'336 \
 --model $model_name \
 --data traffic \
 --features MS \
 --seq_len $seq_len \
 --pred_len 336 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 1 \
 --des 'Exp' \
 --percent 100 \
 --itr 1 --batch_size 16    >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'336'_'100.log

python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_$seq_len'_'720 \
 --model $model_name \
 --data traffic \
 --features MS \
 --seq_len $seq_len \
 --pred_len 720 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 1 \
 --des 'Exp' \
 --percent 100 \
 --itr 1 --batch_size 16   >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'720'_'100.log



python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_$seq_len'_'96 \
 --model $model_name \
 --data traffic \
 --features MS \
 --seq_len $seq_len \
 --pred_len 96 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 1 \
 --des 'Exp' \
 --percent 10 \
 --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'96'_'10.log

python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_$seq_len'_'192 \
 --model $model_name \
 --data traffic \
 --features MS \
 --seq_len $seq_len \
 --pred_len 192 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 1 \
 --des 'Exp' \
 --percent 10 \
 --itr 1 --batch_size 16    >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'192'_'10.log

python -u run_longExp.py \
 --task_name 'long_term_forecast'\
 --is_training 1 \
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_$seq_len'_'336 \
 --model $model_name \
 --data traffic \
 --features MS \
 --seq_len $seq_len \
 --pred_len 336 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 1 \
 --des 'Exp' \
 --percent 10 \
 --itr 1 --batch_size 16    >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'336'_'10.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'720 \
  --model $model_name \
  --data traffic \
  --features MS \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 8   >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'720'_'10.log


