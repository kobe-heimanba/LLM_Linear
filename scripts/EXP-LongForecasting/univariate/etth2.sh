if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi
model_name=LLM_Linear

# ETTh2, univariate results, pred_len= 96 192 336 720

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_96 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTh2_512_96_100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_192 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTh2_512_192_100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_336 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTh2_512_336_100.log


python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_720 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTh2_512_720_100.log



python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_96 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTh2_512_96_10.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_192 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTh2_512_192_10.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_336 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTh2_512_336_10.log


python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_720 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTh2_512_720_10.log

 