from dataset import load_dataset
from train import train, validate
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision.models import resnet50
import torch

weight_decay = 0.01
enable_DRO = False
n_epoch = 300
batch_size = 128


def main():
    train_dataset, val_dataset, test_dataset = load_dataset()
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)

    for n in range(n_epoch):
        print(f"\nEpoch [{n}]")
        print("Training:")
        train(train_data_loader, model, optimizer, enable_DRO)
        print("Validation:")
        validate(val_data_loader, model)

        if (n + 1) % 10 == 0:
            torch.save(model.state_dict(), f'./checkpoints/model_epoch_{n}.pth')
            print(f"Modèle sauvegardé : ./checkpoints/model_epoch_{n}.pth")


if __name__ == '__main__':
    main()