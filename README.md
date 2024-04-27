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
[ViT/Ti-8](https://huggingface.co/gaunernst/vit_tiny_patch8_112.adaface_ms1mv3) |  5,512,640 | MS1MV3 | AdaFace | 96.19% | 99.75% | 97.00% | This repo
[ConvNeXt-Nano](https://huggingface.co/gaunernst/convnext_nano.cosface_ms1mv3) | 15,303,712 | MS1MV3 | CosFace | 97.94% | 99.67% | 97.58% | This repo
[ConvNeXt-Atto](https://huggingface.co/gaunernst/convnext_atto.cosface_ms1mv3) |  3,543,952 | MS1MV3 | CosFace | 96.33% | 99.68% | 96.90% | This repo

TODO: add IJB-B, IJB-C

## Training

Download a dataset from `https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_`

ViT/Ti-8 CosFace. Trained on MS1MV3 for 30 epochs. 12hrs on 1x 4070 Ti SUPER. Change loss to `arcface` or `adaface` to get other variants.

```bash
python train.py --backbone vit_tiny_patch16_224 --backbone_kwargs '{"patch_size":8,"img_size":112}' --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --clip_grad_norm 1 --run_name vit_tiny_cosface --eval_interval 10_000 --loss cosface --compile
```

ConvNeXt-Nano CosFace. Trained on MS1MV3 for 30 epochs.

```bash
python train.py --backbone convnext_nano --backbone_kwargs '{"patch_size":2,"drop_path_rate":0.1}' --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --warmup 0.1 --clip_grad_norm 1 --run_name convnext_nano_cosface --eval_interval 10_000 --loss cosface --channels_last --compile --augmentations "v2.RandomChoice([v2.ColorJitter(0.1,0.1,0.1,0.1), v2.RandomAffine(0,(0.1,0.1))])"
```

ConvNeXt-Atto CosFace. Trained on MS1MV3 for 30 epochs.

```bash
python train.py --backbone convnext_atto --backbone_kwargs '{"patch_size":2}' --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --clip_grad_norm 1 --run_name convnext_atto_cosface --eval_interval 10_000 --loss cosface --channels_last --compile
```

For DDP, replace `python` with `torchrun`. E.g. single-node 4 GPUs

```bash
torchrun --standalone --nproc-per-node=4 train.py --backbone convnext_atto --backbone_kwargs '{"patch_size":2}' --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --clip_grad_norm 1 --run_name convnext_atto_cosface --eval_interval 10_000 --loss cosface --channels_last --compile
```

For more information, see [torchrun](https://pytorch.org/docs/stable/elastic/run.html)

## Notes

Some good papers for training ViT
- MoCov3: https://arxiv.org/abs/2104.02057
- AugReg: https://arxiv.org/pdf/2106.10270
- DeiT3: https://arxiv.org/abs/2204.07118
- Tuned recipe by Lucas Beyer: https://arxiv.org/abs/2205.01580
