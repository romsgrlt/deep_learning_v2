import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import labels



def eval_groups(logits, y, group):
    predicted_classes = torch.argmax(logits, dim=1)
    n_groups = len(labels)

    correct_per_group = torch.zeros(n_groups)
    total_per_group = torch.zeros(n_groups)

    for group_idx in range(n_groups):
        mask = (group == group_idx)
        correct_per_group[group_idx] += (predicted_classes[mask] == y[mask]).sum().item()
        total_per_group[group_idx] += mask.sum().item()

    return correct_per_group, total_per_group


def run(loader, model, criterion, is_training, loss_dro, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train() if is_training else model.eval()

    n_groups = len(labels)

    correct_per_group = torch.zeros(n_groups)
    total_per_group = torch.zeros(n_groups)
    loss_per_group = torch.zeros(n_groups)
    count_per_group = torch.zeros(n_groups)

    with torch.set_grad_enabled(is_training):
        for x, y, group in tqdm(loader):
            x, y, group = x.to(device), y.to(device), group.to(device)

            output = model(x)
            per_sample_losses = criterion(output, y)

            for group_index in range(n_groups):
                mask = (group == group_index)
                if mask.sum() > 0:
                    loss_per_group[group_index] += per_sample_losses[mask].sum().item()
                    count_per_group[group_index] += mask.sum().item()

            if is_training:
                if loss_dro is None:
                    loss = per_sample_losses.mean()
                else:
                    loss = loss_dro.loss(per_sample_losses, group)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            correct, total = eval_groups(output.detach().cpu(), y.cpu(), group.cpu())
            correct_per_group += correct
            total_per_group += total

    group_acc = correct_per_group / total_per_group.clamp(min=1)
    group_loss = loss_per_group / count_per_group.clamp(min=1)

    print(f"  avg_acc={group_acc.mean():.3f}  worst_group_acc={group_acc.min():.3f}")
    for group_index in range(n_groups):
        print(f"    Groupe {group_index} ({labels[group_index]}): acc={group_acc[group_index]:.3f}  loss={group_loss[group_index]:.3f}")

    adv_probs = loss_dro.adv_probs.cpu() if (loss_dro and is_training) else torch.ones(n_groups) / n_groups
    return group_loss, group_acc, adv_probs, total_per_group


def train(data_loader, model, optimizer, loss_dro):
    print("Training:")
    return run(data_loader, model, nn.CrossEntropyLoss(reduction='none'), True, loss_dro, optimizer)


def validate(data_loader, model):
    print('Validate:')
    return run(data_loader, model, nn.CrossEntropyLoss(reduction='none'), False, None)
