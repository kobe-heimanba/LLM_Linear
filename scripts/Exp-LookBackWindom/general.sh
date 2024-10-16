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
seq_len=96
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.001 >logs/LongForecasting/$model_name'_'fS_ETTh1_$seq_len'_'$pred_len'_'100.log
done

seq_len=192
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.001 >logs/LongForecasting/$model_name'_'fS_ETTh1_$seq_len'_'$pred_len'_'100.log
done

seq_len=336
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.001 >logs/LongForecasting/$model_name'_'fS_ETTh1_$seq_len'_'$pred_len'_'100.log
done

seq_len=512
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
  --task_name 'long_term_forecast'\
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --percent 100 \
  --itr 1 --batch_size 16 --feature S --learning_rate 0.001 >logs/LongForecasting/$model_name'_'fS_ETTh1_$seq_len'_'$pred_len'_'100.log
done


