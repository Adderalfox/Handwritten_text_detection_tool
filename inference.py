import torch
from utils.data_loader import get_emnist_dataloader
from models.model import HandwrittenCNN
# from train import load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HandwrittenCNN(num_classes=62)
model.load_state_dict(torch.load('checkpoint40.pth', map_location=device)['model_state_dict'])
model.to(device)
model.eval()

train_loader, val_loader = get_emnist_dataloader()

correct, total = 0, 0
count = 1
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"Tuple: {count}")
        count += 1

print(f'Accuracy on test set: {100 * correct / total: .2f}%')
