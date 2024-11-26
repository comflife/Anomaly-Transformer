export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 3 --num_epochs 10 --batch_size 256 --mode train --dataset TimeSeries --data_path /root/Anomaly-Transformer/dataset/open/train --input_c 26 --output_c 26
# python main.py --anormly_ratio 1  --num_epochs 10        --batch_size 256     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20




