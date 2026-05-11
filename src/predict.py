import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from model import CNN
import os
from scipy import ndimage


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

  # Small, already-cropped inputs should stay close to training distribution.
  if max(img.size) <= 96:
    return img.resize((target_size, target_size), Image.LANCZOS)

  # For phone photos/scans, stretch contrast and normalize polarity.
  img = ImageOps.autocontrast(img, cutoff=2)
  arr = np.array(img)
  if arr.mean() < 127:
    img = ImageOps.invert(img)
    arr = np.array(img)

  # Foreground estimate for document photos.
  fg_thresh = int(np.clip(np.percentile(arr, 18), 40, thresh))
  mask = arr < fg_thresh

  # Keep the largest connected component to suppress page texture/noise.
  if mask.any():
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
      sizes = ndimage.sum(mask, labeled, index=range(1, num_features + 1))
      largest_label = int(np.argmax(sizes) + 1)
      comp_mask = labeled == largest_label
      ys, xs = np.where(comp_mask)
      if ys.size > 0 and xs.size > 0:
        pad_x = max(8, int(0.05 * arr.shape[1]))
        pad_y = max(8, int(0.05 * arr.shape[0]))
        bbox = (
          max(0, xs.min() - pad_x),
          max(0, ys.min() - pad_y),
          min(arr.shape[1], xs.max() + pad_x + 1),
          min(arr.shape[0], ys.max() + pad_y + 1),
        )
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
  with torch.inference_mode():
    image = PREDICT_TRANSFORM(preprocessed).unsqueeze(0).to(device)
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

  confidence_pct = confidence.item() * 100
  top_probabilities, top_indices = torch.topk(probabilities, k=min(3, len(classes)), dim=1)

  predicted_char = classes[predicted_idx.item()]
  top_suggestions = [
    (classes[idx.item()], prob.item() * 100)
    for idx, prob in zip(top_indices[0], top_probabilities[0])
  ]

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