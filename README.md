# Timm Face

Turn any [timm](https://github.com/huggingface/pytorch-image-models) models into a face recognition model.

TODO: evaluation, distillation

## Training

Download a dataset from `https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_`

ViT/Ti-16 CosFace: 92.03% CFP-FP, 99.38% LFW, 92.18% AgeDB-30

```bash
python train.py --backbone vit_tiny_patch16_224 --backbone_kwargs '{"patch_size":8,"img_size":112,"patch_drop_rate":0.2}' --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-4 --weight_decay 1e-2 --run_name cosface_partialfc --eval_interval 10_000 --loss cosface
```
