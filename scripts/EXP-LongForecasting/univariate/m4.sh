if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=LlaLinear


python -u run_longExp.py \
    --task_name 'short_term_forecast' \
    --is_training 1 \
    --root_path ./data/M4/ \
    --data_path Yearly  \
    --model_id Yearly_$seq_len'_'8 \
    --model $model_name \
    --data m4 \
    --features S \
    --seq_len $seq_len \
    --pred_len 8 \
    --enc_in 8 \
    --des 'Exp' \
    --percent 100 \
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'M4_Yearly_$seq_len'_'8'_'100.log

python -u run_longExp.py \
    --task_name 'short_term_forecast' \
    --is_training 1 \
    --root_path ./data/M4/ \
    --data_path Quarterly  \
    --model_id Quarterly_$seq_len'_'8 \
    --model $model_name \
    --data m4 \
    --features S \
    --seq_len $seq_len \
    --pred_len 8 \
    --enc_in 8 \
    --des 'Exp' \
    --percent 100 \
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'M4_Quarterly_$seq_len'_'8'_'100.log

python -u run_longExp.py \
    --task_name 'short_term_forecast' \
    --is_training 1 \
    --root_path ./data/M4/ \
    --data_path Monthly  \
    --model_id Monthly_$seq_len'_'8 \
    --model $model_name \
    --data m4 \
    --features S \
    --seq_len $seq_len \
    --pred_len 8 \
    --enc_in 8 \
    --des 'Exp' \
    --percent 100 \
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'M4_Monthly_$seq_len'_'8'_'100.log

