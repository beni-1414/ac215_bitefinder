import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

root_dir = 'data/bug-bite-images/'
train_dir = root_dir + 'training/'
test_dir = root_dir + 'testing/'

# DenseNet121, EfficientNet-B0, and MobileNetV3-Small
# encoder = models.mobilenetv2(pretrained=True)
# encoder = models.vgg16(pretrained=True)
# encoder = models.densenet169(pretrained=True)
# encoder = models.inception_v3(pretrained=True)
# encoder = models.resnet50(pretrained=True)
encoder = models.resnet18(weights='ResNet18_Weights.DEFAULT')

# Freeze parameters
total_params = len(list(encoder.parameters()))
freeze_prop = 0.75
params_to_freeze = int(freeze_prop * total_params)
for param in list(encoder.parameters())[:params_to_freeze]:
    param.requires_grad = False

class BugBiteClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(BugBiteClassifier, self).__init__()
        self.model = encoder # Image encoder
        self.model.fc = nn.Linear(encoder.fc.in_features, num_classes) # Replace final classification layer

    def forward(self, x):
        return self.model(x)


# Data transforms (see: DeepBiteNet)
resize_dim = 224
transform = transforms.Compose([
    transforms.Resize((resize_dim, resize_dim)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load data
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# Number of labels
num_classes = len(train_dataset.classes)
num_classes

def train(model, optimizer, loss_fn, num_epochs, dataloader, verbose=True):
    losses = []
    accs = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_preds = 0
        total_preds = 0
        for image_X, y in dataloader:
            optimizer.zero_grad()
            y_out = model(image_X)
            loss = loss_fn(y_out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, y_pred = torch.max(y_out, 1)
            correct_preds += (y_pred == y).sum().item()
            total_preds += y.size(0)
        avg_loss = running_loss/total_preds
        avg_acc = correct_preds/total_preds
        if verbose: print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')
        losses.append(avg_loss)
        accs.append(avg_acc)
    return (losses, accs)

model = BugBiteClassifier(encoder, num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

loss, acc = train(model, optimizer, loss_fn, num_epochs, train_dataloader)

def predict(model, dataloader, verbose=True):
    model.eval()
    preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for image_X, y in dataloader:
            y_out = model(image_X)
            _, y_pred = torch.max(y_out, 1)
            preds.extend(y_pred.cpu().numpy())
            correct += (y_pred == y).sum().item()
            total += y.size(0)
    acc = correct/total
    if verbose: print(f'Accuracy: {acc:.4f}')
    return (preds, acc)

preds, acc = predict(model, test_dataloader)