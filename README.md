# Timm Face

Turn any [timm](https://github.com/huggingface/pytorch-image-models) models into a face recognition model.

TODO: evaluation, EMA, distillation

## Training

Download a dataset from `https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_`

Run `train.py`

```bash
python train.py --model vit_tiny_patch16_224 --model_kwargs '{"patch_size":8,"img_size":112}' --ds_path ms1m-retinaface-t1
```
