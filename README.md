# not-cat

This is a small project to train a model to detect `cat` or `not-cat` images.

The goal is to produce a very small model that can be used in resource constrained environments, best case the mode would be `<5MB` in size - but that is a stretch goal so we'll start with `<20MB` and iterate from there based on our learnings.

## Training models

train em

_we'll use `yolo` in the example but we can use `eff` or `yolo_v2` as well_

```bash
uv run not-cat.py train --model eff
# Training mode activated...
# Preloading images into memory...
# 100%|██████████████████████████████████████████████████████████████████| 1250/1250 [00:01<00:00, 1017.94it/s]
# Preloaded 1250 images

# Training on device: cpu
# Epoch 1/5: 100%|████████████████████████████████████████| 5/5 [01:04<00:00, 12.90s/it, loss=0.002, acc=86.8%]
# Epoch 1: Loss=0.002, Acc=86.8%
# Epoch 2/5: 100%|████████████████████████████████████████| 5/5 [00:46<00:00,  9.31s/it, loss=0.001, acc=99.0%]
# Epoch 2: Loss=0.001, Acc=99.0%
# Epoch 3/5: 100%|████████████████████████████████████████| 5/5 [00:47<00:00,  9.41s/it, loss=0.000, acc=99.3%]
# Epoch 3: Loss=0.000, Acc=99.3%
# Epoch 4/5: 100%|████████████████████████████████████████| 5/5 [00:47<00:00,  9.42s/it, loss=0.000, acc=99.7%]
# Epoch 4: Loss=0.000, Acc=99.7%
# Epoch 5/5: 100%|████████████████████████████████████████| 5/5 [00:47<00:00,  9.44s/it, loss=0.000, acc=99.6%]
# Epoch 5: Loss=0.000, Acc=99.6%
```

## Check performance

of course we can look at the training logs, but we tbh we just want a quick vibe check, so we'll feed in a couple of images and see what the model predicts.

```bash
uv run not-cat.py run --model eff
# Running mode activated...
# rock-climb:	not cat
# cute-cat-p:	cat
```

## Exported Models

| Model                              | Size |
| ---------------------------------- | ---- |
| eff_backbone_classifier.pth        | 16M  |
| yolov8n_backbone_classifier.pth    | 639K |
| yolov8n_v2_backbone_classifier.pth | 4.4M |

## Thoughts and TODO

- the efficientnet models train well and perform well, but a bit bigger than we'd like (can we reduce and still maintain performance?)
- the yolo models train well, but the performance is not great (we should revisit the arch and data)
- we export onnx models but are not using them yet (add a small server that loads the models and serves them)
- although we're targeting cpu can we quantize in a helpful way? (we should look into this)
- add script for fetching data (from kaggle datasets)
- improve examples with pictures of my cats! 🐱
- upload final models to huggingface for easy access (add script for this)
