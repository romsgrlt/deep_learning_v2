from dataset import load_dataset
from train import train, validate
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision.models import resnet50
import torch
from dro import GroupDROLoss
from logger import CSVLogger
from dataset import labels

weight_decay = 0.01
enable_DRO = True
n_epoch = 300
batch_size = 128


def log_row(logger, epoch, group_loss, group_acc, adv_probs, group_total):
    row = {'epoch': epoch}
    for g in range(len(labels)):
        row[f'loss_group_{labels[g]}'] = round(group_loss[g].item(), 6)
        row[f'accuracy_group_{labels[g]}'] = round(group_acc[g].item(), 6)
        row[f'adv_prob_group_{labels[g]}'] = round(adv_probs[g].item(), 6)
        row[f'n_{labels[g]}'] = int(group_total[g].item())
    row['avg_loss'] = round(group_loss.mean().item(), 6)
    row['avg_accuracy'] = round(group_acc.mean().item(), 6)
    row['worst_group_accuracy'] = round(group_acc.min().item(), 6)
    logger.log(row)


def main():
    train_dataset, val_dataset, test_dataset = load_dataset()
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)

    dro_loss = GroupDROLoss(n_groups=4).to(device) if enable_DRO else None

    train_logger = CSVLogger('./logs/train.csv', len(labels))
    val_logger = CSVLogger('./logs/val.csv', len(labels))

    for n in range(n_epoch):
        print(f"\nEpoch [{n}]")
        print("Training:")
        group_loss, group_acc, adv_probs, group_total = train(train_data_loader, model, optimizer, dro_loss)
        log_row(train_logger, n, group_loss, group_acc, adv_probs, group_total)

        print("Validation:")
        group_loss, group_acc, adv_probs, group_total = validate(val_data_loader, model)
        log_row(val_logger, n, group_loss, group_acc, adv_probs, group_total)

        if (n + 1) % 10 == 0:
            train_logger.flush()
            val_logger.flush()
            torch.save(model.state_dict(), f'./checkpoints/model_epoch_{n}.pth')
            print(f"Modèle sauvegardé : ./checkpoints/model_epoch_{n}.pth")

    train_logger.close()
    val_logger.close()


if __name__ == '__main__':
    main()
