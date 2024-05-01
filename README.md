# Timm Face

Turn any [timm](https://github.com/huggingface/pytorch-image-models) models into a face recognition model.

TODO: distillation

## Comparison

For CFP-FP, LFW, AgeDB-30, metric is 10-fold accuracy. For IJB-B and IJB-C, metric is TAR@FAR=1e-4

Model       | Params     | Dataset    | Loss    | CFP-FP | LFW    | AgeDB-30 | IJB-B  | IJB-C  | Source
------------|------------|------------|---------|--------|--------|----------|--------|--------|-------
iResNet-100 | 65,150,912 | WebFace12M | AdaFace | 99.30% | 99.82% | 97.95%   | 96.23% | 97.49% | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)
iResNet-100 | 65,156,288 | Glint360k  | CosFace(?) | 99.16% | 99.83% | 98.45% | 95.78% | 97.04% | [deepinsight/insightface](https://github.com/deepinsight/insightface) (antelopev2)
iResNet-50  | 43,590,976 | WebFace12M | CosFace(?) | 99.24% | 99.80% | 98.07% | 95.35% | 96.83% | [deepinsight/insightface](https://github.com/deepinsight/insightface) (buffalo_l)
iResNet-18  | 24,020,352 | WebFace4M  | AdaFace | 97.53% | 99.60% | 96.33%   | 92.00% | 94.16% | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)
GhostFaceNetV1 (S1) | 4,087,794 | MS1MV3 | ArcFace | 97.01% | 99.52% | 97.12% | 92.48% | 94.46% | [HamadYA/GhostFaceNets](https://github.com/HamadYA/GhostFaceNets)
[ViT/S-8 (GAP)](https://huggingface.co/gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3) | 21,640,832 | MS1MV3 | CosFace | 98.56% | 99.78% | 97.90% | 94.38% | 95.88% | This repo
[ViT/Ti-8](https://huggingface.co/gaunernst/vit_tiny_patch8_112.cosface_ms1mv3) |  5,512,640 | MS1MV3 | CosFace | 96.44% | 99.77% | 97.23% | 92.69% | 94.49% | This repo
[ViT/Ti-8](https://huggingface.co/gaunernst/vit_tiny_patch8_112.arcface_ms1mv3) |  5,512,640 | MS1MV3 | ArcFace | 96.91% | 99.67% | 97.17% | 91.78% | 93.63% | This repo
[ViT/Ti-8](https://huggingface.co/gaunernst/vit_tiny_patch8_112.adaface_ms1mv3) |  5,512,640 | MS1MV3 | AdaFace | 96.19% | 99.75% | 97.00% | 91.95% | 93.81% | This repo
[ConvNeXt-Nano](https://huggingface.co/gaunernst/convnext_nano.cosface_ms1mv3)  | 15,303,712 | MS1MV3 | CosFace | 97.94% | 99.67% | 97.58% | 93.45% | 95.13% | This repo
[ConvNeXt-Atto](https://huggingface.co/gaunernst/convnext_atto.cosface_ms1mv3)  |  3,543,952 | MS1MV3 | CosFace | 96.33% | 99.68% | 96.90% | 91.76% | 93.58% | This repo

NOTE:
- For mk-minchul/AdaFace models, input image is BGR.
- For GhostFaceNet, I export the TensorFlow model to ONNX and run inference with the ONNX model.

## Training

Download a dataset from `https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_`

ViT/S-8 (GAP) CosFace.

```bash
python train.py --backbone vit_small_patch16_224 --backbone_kwargs '{"patch_size":8,"img_size":112,"patch_drop_rate":0.1,"class_token":false,"global_pool":"avg"}' --ds_path ms1m-retinaface-t1 --batch_size 384 --total_steps 400_000 --lr 5e-4 --weight_decay 5e-2 --clip_grad_norm 1 --run_name vit_small_gap_cosface_randaugment --eval_interval 10_000 --loss cosface --compile --augmentations "v2.RandAugment()"
```

ViT/Ti-8 CosFace. Trained on MS1MV3 for 30 epochs. 12hrs on 1x 4070 Ti SUPER. Change loss to `arcface` or `adaface` to get other variants.

```bash
python train.py --backbone vit_tiny_patch16_224 --backbone_kwargs '{"patch_size":8,"img_size":112}' --partial_fc 16_384 --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --clip_grad_norm 1 --run_name vit_tiny_cosface --eval_interval 10_000 --loss cosface --compile
```

ConvNeXt-Nano CosFace. Trained on MS1MV3 for 30 epochs.

```bash
python train.py --backbone convnext_nano --backbone_kwargs '{"patch_size":2,"drop_path_rate":0.1}' --partial_fc 16_384 --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --warmup 0.1 --clip_grad_norm 1 --run_name convnext_nano_cosface --eval_interval 10_000 --loss cosface --channels_last --compile --augmentations "v2.RandomChoice([v2.ColorJitter(0.1,0.1,0.1,0.1), v2.RandomAffine(0,(0.1,0.1))])"
```

ConvNeXt-Atto CosFace. Trained on MS1MV3 for 30 epochs.

```bash
python train.py --backbone convnext_atto --backbone_kwargs '{"patch_size":2}' --partial_fc 16_384 --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --clip_grad_norm 1 --run_name convnext_atto_cosface --eval_interval 10_000 --loss cosface --channels_last --compile
```

For DDP, replace `python` with `torchrun`. E.g. single-node 4 GPUs

```bash
torchrun --standalone --nproc-per-node=4 train.py --backbone convnext_atto --backbone_kwargs '{"patch_size":2}' --ds_path ms1m-retinaface-t1 --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --clip_grad_norm 1 --run_name convnext_atto_cosface --eval_interval 10_000 --loss cosface --channels_last --compile
```

For more information, see [torchrun](https://pytorch.org/docs/stable/elastic/run.html)

## Other train options

TPU training (using [torch-xla](https://github.com/pytorch/xla)) (WIP)

```bash
python train_tpu.py --backbone convnext_atto --backbone_kwargs '{"patch_size":2}' --ds_path "https://huggingface.co/datasets/gaunernst/ms1mv3-wds/resolve/main/ms1mv3-{0000..0099}.tar" --batch_size 768 --total_steps 200_000 --lr 1e-3 --weight_decay 1e-1 --clip_grad_norm 1 --run_name convnext_atto_cosface --eval_interval 10_000 --loss cosface
```

## Evaluation

For IJB-B and IJB-C, download dataset and metadata from `https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/ijb`. Then run the following command

```bash
python test_ijb.py --dataset IJBC --model "hf_hub:gaunernst/vit_tiny_patch8_112.cosface_ms1mv3" --model_kwargs '{"pretrained":true}'
```

## Notes

Some good papers for training ViT
- MoCov3: https://arxiv.org/abs/2104.02057
- AugReg: https://arxiv.org/pdf/2106.10270
- DeiT3: https://arxiv.org/abs/2204.07118
- Tuned recipe by Lucas Beyer: https://arxiv.org/abs/2205.01580
