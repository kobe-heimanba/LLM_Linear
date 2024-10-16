if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi
model_name=LlaLinear

# ETTm1, univariate results, pred_len= 24 48 96 192 336 720
#python -u run_longExp.py \
#  --task_name 'long_term_forecast'\
#  --is_training 1 \
#  --root_path ./data/ETT/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_512_24 \
#  --model $model_name \
#  --data ETTm1 \
#  --seq_len 512 \
#  --pred_len 24 \
#  --enc_in 1 \
#  --des 'Exp' \
#  --percent 100\
#  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_336_24.log
#
#python -u run_longExp.py \
#  --task_name 'long_term_forecast'\
#  --is_training 1 \
#  --root_path ./data/ETT/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_512_48 \
#  --model $model_name \
#  --data ETTm1 \
#  --seq_len 512 \
#  --pred_len 48 \
#  --enc_in 1 \
#  --des 'Exp' \
#  --percent 100\
#  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_336_48.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_96_100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_192 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_192_100.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_336 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_336_100.log


python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_720 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_720_100.log


# ETTm1, univariate results, pred_len= 24 48 96 192 336 720
# python -u run_longExp.py \
#   --task_name 'long_term_forecast'\
#   --is_training 1 \
#   --root_path ./data/ETT/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_512_24 \
#   --model $model_name \
#   --data ETTm1 \
#   --seq_len 512 \
#   --pred_len 24 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --percent 10\
#   --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_336_24.log

# python -u run_longExp.py \
#   --task_name 'long_term_forecast'\
#   --is_training 1 \
#   --root_path ./data/ETT/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_512_48 \
#   --model $model_name \
#   --data ETTm1 \
#   --seq_len 512 \
#   --pred_len 48 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --percent 10\
#   --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_336_48.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_96_10.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_192 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_192_10.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_336 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_336_10.log


python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_720 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 10 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_720_10.log

  # ETTm1, univariate results, pred_len= 24 48 96 192 336 720
# python -u run_longExp.py \
#   --task_name 'long_term_forecast'\
#   --is_training 1 \
#   --root_path ./data/ETT/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_512_24 \
#   --model $model_name \
#   --data ETTm1 \
#   --seq_len 512 \
#   --pred_len 24 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --percent 5\
#   --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_336_24.log
 
# python -u run_longExp.py \
#   --task_name 'long_term_forecast'\
#   --is_training 1 \
#   --root_path ./data/ETT/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_512_48 \
#   --model $model_name \
#   --data ETTm1 \
#   --seq_len 512 \
#   --pred_len 48 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --percent 5\
#   --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_336_48.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 5 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_96_5.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_192 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 5 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_192_5.log

python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_336 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 5 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_336_5.log


python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_720 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --percent 5 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'fS_ETTm1_512_720_5.log