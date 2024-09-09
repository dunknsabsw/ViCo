CUDA_LAUNCH_BLOCKING=1.

cap_list=(1 0.3)
caplen_list=(384 256 128 64 32 16)
e_list=(10)

# cap_list=(1 0.3)
# caplen_list=(384 256 128)
# e_list=(10)

for cap in ${cap_list[@]}; do
for e in ${e_list[@]}; do
for caplen in ${caplen_list[@]}; do

exp_id=caplen${caplen}_e${e}_cap${cap}
# exp_id=debug

torchrun \
--rdzv_endpoint 127.0.0.1:1234 \
--nproc_per_node 8 \
/root/cj/cap0825/train.py \
--model 7B \
--max_seq_len 128 \
--max_cap_len ${caplen} \
--batch_size 4 \
--epochs ${e} \
--warmup_epochs 2 \
--bias 3.5 \
--tau 100. \
--max_feats 10 \
--dataset intentqa \
--blr 9e-2 \
--weight_decay 0.14 \
--output_dir ./output/intentqa_7b_random_cap/${exp_id} \
--accum_iter 2 \
--cap ${cap} \
#--init_eval \

done
done
done
