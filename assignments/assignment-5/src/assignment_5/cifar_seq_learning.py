import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

def get_data_loaders(dataset_name, batch_size=512):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    else:
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader, num_classes


def get_model(num_classes, pretrained=False):
    model = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# Training Function
def train_model(model, train_loader, test_loader, epochs=100, lr=0.001, freeze_features=False):
    model = model.to(device)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device=device)

    for epoch in range(epochs):
        # TRAINING
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # TESTING
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        test_loss /= len(test_loader)
        scheduler.step()

        # LOGGING
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d}: Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")

    return test_acc

# Main Experiment
def main():
    batch_size = 512
    epochs = 100
    lr = 0.001

    # EXPERIMENT A
    print("EXPERIMENT A: CIFAR-100 -> CIFAR-10")
    wandb.init(project="Q4-Cifar-Sequential-Exp", name="Experiment_A_CIFAR100_to_CIFAR10")

    # Phase 1: Train on CIFAR-100
    train_loader, test_loader, _ = get_data_loaders('CIFAR100', batch_size)
    model = get_model(num_classes=100, pretrained=True)
    wandb.log({'phase': 1, 'dataset': 'CIFAR-100'})
    cifar100_acc = train_model(model, train_loader, test_loader, epochs, lr, freeze_features=False)

    # Phase 2: Transfer Learning on CIFAR-10
    print("Transfer learning to CIFAR-10...")
    train_loader, test_loader, _ = get_data_loaders('CIFAR10', batch_size)
    model.classifier[1] = nn.Linear(model.last_channel, 10).to(device)
    wandb.log({'phase': 2, 'dataset': 'CIFAR-10'})
    cifar10_acc_A = train_model(model, train_loader, test_loader, epochs, lr, freeze_features=True)

    wandb.log({'final_cifar100_accuracy': cifar100_acc, 'final_cifar10_accuracy': cifar10_acc_A})
    wandb.finish()

    # EXPERIMENT B
    print("EXPERIMENT B: CIFAR-10 -> CIFAR-100")
    wandb.init(project="Q4-Cifar-Sequential-Exp", name="Experiment_B_CIFAR10_to_CIFAR100_b")

    # Phase 1: Train on CIFAR-10
    train_loader, test_loader, _ = get_data_loaders('CIFAR10', batch_size)
    model = get_model(num_classes=10, pretrained=False)
    wandb.log({'phase': 1, 'dataset': 'CIFAR-10'})
    cifar10_acc = train_model(model, train_loader, test_loader, epochs, lr, freeze_features=False)

    # Phase 2: Transfer Learning on CIFAR-100
    print("Transfer learning to CIFAR-100...")
    train_loader, test_loader, _ = get_data_loaders('CIFAR100', batch_size)
    model.classifier[1] = nn.Linear(model.last_channel, 100).to(device)
    wandb.log({'phase': 2, 'dataset': 'CIFAR-100'})
    cifar100_acc_B = train_model(model, train_loader, test_loader, epochs, lr, freeze_features=True)

    wandb.log({'final_cifar10_accuracy': cifar10_acc, 'final_cifar100_accuracy': cifar100_acc_B})
    wandb.finish()

    # RESULTS
    print("RESULTS: ")
    print(f"Experiment A (CIFAR-100 -> CIFAR-10):")
    print(f"  CIFAR-100 final accuracy: {cifar100_acc:.2f}%")
    print(f"  CIFAR-10 final accuracy:  {cifar10_acc_A:.2f}%")
    print(f"Experiment B (CIFAR-10 -> CIFAR-100):")
    print(f"  CIFAR-10 final accuracy:  {cifar10_acc:.2f}%")
    print(f"  CIFAR-100 final accuracy: {cifar100_acc_B:.2f}%")

if __name__ == "__main__":
    main()
