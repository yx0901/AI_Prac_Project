import torch
from PIL import Image
from torchvision import transforms
from model import CNN
import os

def predict(image_path, model_path='models/cnn.pth'):
    # Check if image file exists
    if not os.path.exists(image_path):
      print(f"Error: Cannot find file '{image_path}'")
      return None, None

    # Check if model file exists
    if not os.path.exists(model_path):
      print(f"Error: Cannot find model file")
      return None, None

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = checkpoint['classes']
    
    model = CNN(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Preprocess image with deterministic evaluation-style transforms.
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        top_probabilities, top_indices = torch.topk(probabilities, k=min(3, len(classes)), dim=1)
    
    predicted_char = classes[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    top_suggestions = [
        (classes[idx.item()], prob.item() * 100)
        for idx, prob in zip(top_indices[0], top_probabilities[0])
    ]

    print("Top 3 suggestions:")
    for rank, (label, score) in enumerate(top_suggestions, start=1):
        print(f"  {rank}. {label} ({score:.1f}% confidence)")

    if confidence_pct < 60:
      print(f"Not quite sure but this might be a {predicted_char} ({confidence_pct:.1f}% confidence)")
      return predicted_char, confidence_pct, top_suggestions
    else:
      print(f"Predicted: {predicted_char} ({confidence_pct:.1f}% confidence)")
      return predicted_char, confidence_pct, top_suggestions

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <image_path>")