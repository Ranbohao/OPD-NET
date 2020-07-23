#!/bin/bash
echo "start train !"
start_t=$(date +%s)
python train_new.py --gpu=1
end_t=$(date +%s)
time_t=$(( $end_t - $start_t ))

echo "start infer !"
start=$(date +%s)
python inference.py --gpu=1
end=$(date +%s)
time=$(( $end - $start ))
echo "eval total time cost :"
echo $time
echo "train total time cost :"
echo $time_t
python eval.py
