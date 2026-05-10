# Chinese Handwriting Recognition

A deep learning system for recognizing handwritten Chinese characters using PyTorch. This project implements a CNN-based approach to classify individual handwritten characters from images.

## References
- Kaggle Dataset: https://www.kaggle.com/datasets/gpreda/chinese-mnist/data

### First install PyTorch and other dependencies:
pip install -r requirements.txt

## Commands
**Train** 
```bash
python src/train.py
```
or 
(with checkpointing, LR scheduling, and early stopping):
```bash
python src/train.py --epochs 20 --batch-size 64 --learning-rate 1e-3 --patience 5
```

**Resume** from the latest best checkpoint:
```bash
python src/train.py --resume
```

**Evaluate** (top-1 and top-3 accuracy with per-class breakdown):
```bash
python3 src/evaluate.py
```
```bash
python src/evaluate.py --model models/cnn.pth --test-dir data/test --batch-size 32
```

**Predict** (top-3 suggestions with confidence scores):
```bash
python src/predict.py data/user_input/0.png
```

## Checkpoints
Checkpoints are saved to `models/cnn.pth` with per-epoch snapshots. Each checkpoint includes the model, optimizer, and scheduler state, class mappings, and best validation accuracy. Use `--resume` to continue training from the latest checkpoint.