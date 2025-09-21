#!/bin/bash

python main.py train --lr=3e-5 --augmentation="aggressive" --weight_decay=1e-3 --composite=True --regions=None --lrcn=False --model="DenseNet121" --batch_size=10 --gpus=1 --patience=25 --pretrained=True --load_aux=True --late_fusion_regions='onehot' --late_fusion_polygon_loss=True --gpus=1

## past examples

# python main.py train --lr=3e-5 --augmentation="aggressive" --weight_decay=1e-3 --composite=False --regions=None --lrcn=True --model="Sequential2DClassifier-DenseNet121" --gpus=1 --batch_size=10 --patience=25 --pretrained=True

# python main.py train --exp_name="train_AS" --regions=['AS'] --model="DenseNet121" --gpus=1

