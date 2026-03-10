# Chinese Handwriting Recognition

A deep learning system for recognizing handwritten Chinese characters using PyTorch. This project implements a CNN-based approach to classify individual handwritten characters from images.

## References
- Kaggle Dataset: https://www.kaggle.com/datasets/gpreda/chinese-mnist/data

### First install PyTorch and other dependencies:
pip install -r requirements.txt

### To train the model:
python src/train.py

### To predict on a new image example:
python src/predict.py data/user_input/0.png