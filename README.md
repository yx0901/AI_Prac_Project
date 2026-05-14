# Chinese Handwriting Recognition

A deep learning system for recognizing handwritten Chinese characters using PyTorch. This project implements a CNN-based approach to classify individual handwritten characters from images.

## References
- Kaggle Dataset: https://www.kaggle.com/datasets/vitaliikyzym/chinese-handwriting-recognition-hsk-1 

### First install PyTorch and other dependencies:
pip install -r requirements.txt

If you want to use the popup drawing board tool, install tkinter as a system package on Linux:
```bash
sudo apt-get install python3-tk
```

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

**Draw and Predict** (popup blackboard input):
```bash
python3 src/draw_predict.py
```
Use the Pen size slider for thinner strokes, then click Predict.

Run without arguments to keep the model loaded and predict multiple images in one session:
```bash
python3 src/predict.py
```

Example session:
```text
Model loaded. Enter an image path, or type 'quit' to exit.
Image path> data/user_input/100.jpg
Image path> quit
```

## Checkpoints
Checkpoints are saved to `models/cnn.pth` with per-epoch snapshots. Each checkpoint includes the model, optimizer, and scheduler state, class mappings, and best validation accuracy. Use `--resume` to continue training from the latest checkpoint.