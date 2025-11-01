model_name=TimeLLM
llm_model='GPT2'
train_epochs=1
learning_rate=0.01
llama_layers=12

master_port=00097
num_process=1
batch_size=24
d_model=32
d_ff=128
llm_dim=768

comment='TimeLLM-FREDd2'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path FREDd2.csv \
  --model_id FREDd2_128_12 \
  --model $model_name \
  --data FREDd2 \
  --features MS \
  --seq_len 128 \
  --label_len 24 \
  --pred_len 12 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --llm_dim $llm_dim

