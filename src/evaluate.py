import argparse
import torch
from torch.utils.data import DataLoader

from dataset import ChineseCharDataset
from model import CNN


def evaluate(model_path='models/cnn.pth', test_dir='data/test', batch_size=32):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load checkpoint from training.
	checkpoint = torch.load(model_path, map_location=device)
	classes = checkpoint['classes']
	class_to_idx = checkpoint['class_to_idx']

	model = CNN(num_classes=len(classes)).to(device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	test_dataset = ChineseCharDataset(root_dir=test_dir)

	# Build label remapping from test dataset indices to training indices.
	missing_classes = [
		c for c in test_dataset.classes if c not in class_to_idx
	]
	if missing_classes:
		raise ValueError(
			f"Test set contains unseen classes not in checkpoint: {missing_classes}"
		)

	test_to_train_idx = {
		test_idx: class_to_idx[test_dataset.idx_to_class[test_idx]]
		for test_idx in test_dataset.idx_to_class
	}

	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	total = 0
	correct = 0

	per_class_total = {c: 0 for c in classes}
	per_class_correct = {c: 0 for c in classes}

	with torch.no_grad():
		for images, labels in test_loader:
			images = images.to(device)
			mapped_labels = torch.tensor(
				[test_to_train_idx[label.item()] for label in labels],
				dtype=torch.long,
				device=device,
			)

			outputs = model(images)
			preds = torch.argmax(outputs, dim=1)

			total += mapped_labels.size(0)
			correct += (preds == mapped_labels).sum().item()

			for i in range(mapped_labels.size(0)):
				class_name = classes[mapped_labels[i].item()]
				per_class_total[class_name] += 1
				if preds[i].item() == mapped_labels[i].item():
					per_class_correct[class_name] += 1

	if total == 0:
		raise ValueError(f"No test images found in '{test_dir}'.")

	accuracy = 100.0 * correct / total
	print(f"Test samples: {total}")
	print(f"Test accuracy: {accuracy:.2f}%")

	print("\nPer-class accuracy:")
	for class_name in classes:
		class_total = per_class_total[class_name]
		class_correct = per_class_correct[class_name]
		if class_total == 0:
			print(f"  {class_name}: N/A (no samples)")
			continue
		class_acc = 100.0 * class_correct / class_total
		print(f"  {class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
	parser.add_argument('--model', default='models/cnn.pth', help='Path to model checkpoint')
	parser.add_argument('--test-dir', default='data/test', help='Path to test dataset root')
	parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')

	args = parser.parse_args()
	evaluate(model_path=args.model, test_dir=args.test_dir, batch_size=args.batch_size)
