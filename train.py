from torch import optim
from models.model import HandwrittenCNN
from utils.data_loader import get_emnist_dataloader
import torch.nn as nn
import torch

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    return model, optimizer, start_epoch, loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandwrittenCNN().to(device)

# checkpoint_path = 'handwritten_cnn_epoch50.pth'
# model.load_state_dict(torch.load(checkpoint_path))
# print(f"Resumed from checkpoint: {checkpoint_path}")

train_loader, val_loader = get_emnist_dataloader()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training has started")
for epoch in range(50):
    model.train()
    for images, labels in train_loader:
        images, labels  = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        # torch.save(model.state_dict(), f'handwritten_cnn_epoch{epoch+1}.pth')
        save_checkpoint(model, optimizer, epoch, loss, f'checkpoint{epoch + 1}.pth')
        print(f"Checkpoint saved: checkpoint{epoch+1}.pth")

    print(f'Epoch {epoch+1} complete. Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'handwritten_cnn.pth')
print('Model saved!')