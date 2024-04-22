# Timm Face

Turn any [timm](https://github.com/huggingface/pytorch-image-models) models into a face recognition model.

TODO: distillation

## Comparison

Model       | Params     | Dataset    | Loss    | CFP-FP | LFW    | AgeDB-30 | Source
------------|------------|------------|---------|--------|--------|----------|-------
iResNet-101 | 65,150,912 | WebFace12M | AdaFace | 99.21% | 99.82% | 98.00%   | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)
iResNet-18  | 24,020,352 | WebFace4M  | AdaFace | 97.06% | 99.50% | 96.25%   | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)
GhostFaceNetV1 (S1) | 4,088,812 | MS1MV3 | ArcFace | 97.06% | 99.53% | 97.13% | [HamadYA/GhostFaceNets](https://github.com/HamadYA/GhostFaceNets)
[ViT/Ti-8](https://huggingface.co/gaunernst/vit_tiny_patch8_112.cosface_ms1mv3) |  5,512,640 | MS1MV3 | CosFace | 96.44% | 99.77% | 97.23% | This repo
[ViT/Ti-8](https://huggingface.co/gaunernst/vit_tiny_patch8_112.arcface_ms1mv3) |  5,512,640 | MS1MV3 | ArcFace | 96.91% | 99.67% | 97.17% | This repo

TODO: add IJB-B, IJB-C

## Training

Download a dataset from `https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_`

ViT/Ti-8 CosFace. Trained on MS1MV3 for 30 epochs. 12hrs on 1x 4070 Ti SUPER.

```bash
python train.py --backbone vit_tiny_patch16_224 --backbone_kwargs '{"patch_size":8,"img_size":112}' --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --clip_grad_norm 1 --run_name vit_tiny_cosface --eval_interval 10_000 --loss cosface --compile
```
