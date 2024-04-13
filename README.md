# Timm Face

Turn any [timm](https://github.com/huggingface/pytorch-image-models) models into a face recognition model.

TODO: distillation

## Comparison

Model      | Params     | Dataset    | Loss    | CFP-FP | LFW    | AgeDB-30 | Note
-----------|------------|------------|---------|--------|--------|----------|-----
ResNet-101 | 65,150,912 | WebFace12M | AdaFace | 99.21% | 99.82% | 98.00%   | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)
ResNet-18  | 24,020,352 | WebFace4M  | AdaFace | 97.06% | 99.50% | 96.25%   | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)
ViT/Ti-8   |  5,512,256 | MS1MV3     | CosFace | 94.29% | 99.53% | 94.50%   | This repo (trained from scratch)

## Training

Download a dataset from `https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_`

ViT/Ti-8 CosFace: 94.21% CFP-FP, 99.53% LFW, 94.32% AgeDB-30. 10hrs on 1x 4070 Ti SUPER.

```bash
python train.py --backbone vit_tiny_patch16_224 --backbone_kwargs '{"patch_size":8,"img_size":112,"patch_drop_rate":0.2,"class_token":false,"global_pool":"avg"}' --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-2 --run_name cosface_partialfc_gap --eval_interval 10_000 --loss cosface --compile
```
