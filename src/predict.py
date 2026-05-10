import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from model import CNN
import os


PREDICT_TRANSFORM = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5])
])


def load_model(model_path, device):
  checkpoint = torch.load(model_path, map_location=device)
  classes = checkpoint['classes']

  model = CNN(num_classes=len(classes)).to(device)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  return model, classes

def preprocess_user_image(pil_img, target_size=64, thresh=200):
  img = pil_img.convert('L')
  arr = np.array(img)

  # If background is dark, invert so strokes are dark on light background
  if arr.mean() < 127:
    img = ImageOps.invert(img)
    arr = np.array(img)
  # Determine ink vs background polarity robustly (invert only if needed)
  mask = arr < thresh
  training_strokes_are_bright = False
  try:
    # auto-detect training polarity from a representative training image
    train_ds = ChineseCharDataset(root_dir='data/train')
    if train_ds.samples:
      sample_path, _ = train_ds.samples[0]
      sample = Image.open(sample_path).convert('L')
      s_arr = np.array(sample)
      s_mask = s_arr < thresh
      if s_mask.any():
        s_ink_mean = s_arr[s_mask].mean()
        s_bg_mean = s_arr[~s_mask].mean()
        training_strokes_are_bright = bool(s_ink_mean > s_bg_mean)
  except Exception:
    # fallback: assume strokes are dark on light background (common)
    training_strokes_are_bright = False

  if mask.any():
    ink_mean = arr[mask].mean()
    bg_mean = arr[~mask].mean()
    strokes_are_bright = bool(ink_mean > bg_mean)
  else:
    strokes_are_bright = arr.mean() > 127

  if strokes_are_bright != training_strokes_are_bright:
    img = ImageOps.invert(img)
    arr = np.array(img)

  # Find ink bbox
  mask = arr < thresh
  if mask.any():
    ys, xs = np.where(mask)
    pad = 10
    bbox = (max(0, xs.min()-pad), max(0, ys.min()-pad), 
        min(arr.shape[1], xs.max()+pad+1), min(arr.shape[0], ys.max()+pad+1))
    img = img.crop(bbox)

  # Pad to square and center
  w, h = img.size
  size = max(w, h)
  canvas = Image.new('L', (size, size), 255)
  paste_x = (size - w) // 2
  paste_y = (size - h) // 2
  canvas.paste(img, (paste_x, paste_y))

  out = canvas.resize((target_size, target_size), Image.LANCZOS)
  # Save preprocessed image for inspection
  try:
    out.convert('RGB').save('tmp_preprocessed.png')
  except Exception:
    pass
  return out


def predict(image_path, model, classes, device):
  if not os.path.exists(image_path):
    print(f"Error: Cannot find file '{image_path}'")
    return None, None, None

  raw = Image.open(image_path)
  preprocessed = preprocess_user_image(raw, target_size=64)
  candidates = [
    ("original", preprocessed),
    ("inverted", ImageOps.invert(preprocessed)),
  ]

  best = None
  with torch.inference_mode():
    for polarity_name, candidate_img in candidates:
      image = PREDICT_TRANSFORM(candidate_img).unsqueeze(0).to(device)
      outputs = model(image)
      probabilities = torch.softmax(outputs, dim=1)
      confidence, predicted_idx = torch.max(probabilities, 1)
      if best is None or confidence.item() > best["confidence"]:
        best = {
          "polarity": polarity_name,
          "confidence": confidence.item(),
          "predicted_idx": predicted_idx.item(),
          "probabilities": probabilities,
        }

  probabilities = best["probabilities"]
  predicted_idx = best["predicted_idx"]
  confidence_pct = best["confidence"] * 100
  top_probabilities, top_indices = torch.topk(probabilities, k=min(3, len(classes)), dim=1)

  predicted_char = classes[predicted_idx]
  top_suggestions = [
    (classes[idx.item()], prob.item() * 100)
    for idx, prob in zip(top_indices[0], top_probabilities[0])
  ]

  print(f"Using best polarity path: {best['polarity']}")
  print("Top 3 suggestions:")
  for rank, (label, score) in enumerate(top_suggestions, start=1):
    print(f"  {rank}. {label} ({score:.1f}% confidence)")

  if confidence_pct < 60:
    print(f"Not quite sure but this might be a {predicted_char} ({confidence_pct:.1f}% confidence)")
  else:
    print(f"Predicted: {predicted_char} ({confidence_pct:.1f}% confidence)")

  return predicted_char, confidence_pct, top_suggestions


def interactive_predict(model_path='models/cnn.pth'):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if not os.path.exists(model_path):
    print("Error: Cannot find model file")
    return

  model, classes = load_model(model_path, device)
  print("Model loaded. Enter an image path, or type 'quit' to exit.")

  while True:
    image_path = input("Image path> ").strip()
    if image_path.lower() in {'quit', 'exit', 'q'}:
      break
    if not image_path:
      continue
    predict(image_path, model, classes, device)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = sys.argv[2]
        if not os.path.exists(model_path):
            print("Error: Cannot find model file")
        else:
            model, classes = load_model(model_path, device)
            predict(sys.argv[1], model, classes, device)
    elif len(sys.argv) > 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = 'models/cnn.pth'
        if not os.path.exists(model_path):
            print("Error: Cannot find model file")
        else:
            model, classes = load_model(model_path, device)
            predict(sys.argv[1], model, classes, device)
    else:
        interactive_predict()