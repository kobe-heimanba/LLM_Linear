
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=LLM_Linear
#batch_size 16
python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 24 \
  --enc_in 7 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'24'_'100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'36 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 36 \
  --enc_in 7 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'36'_'100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'48 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 48 \
  --enc_in 7 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'48'_'100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'60 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 60 \
  --enc_in 7 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'60'_'100.log

