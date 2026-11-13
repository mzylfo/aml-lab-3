import os
import argparse
import torch
from torch import nn
import wandb

# Importiamo i nostri moduli creati nei passi precedenti!
from models import CustomNet
from dataset import get_dataloaders
from utils import visualize_samples
from eval import validate


def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.4f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy

def main(args):
    # 1. Imposta il device (GPU se disponibile, altrimenti CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Inizializza WandB e salva gli iperparametri
    wandb.init(project=args.wandb_project, config=vars(args))

    # 3. Carica i DataLoader
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 4. Logga la griglia di visualizzazione su WandB (una volta all'inizio)
    fig = visualize_samples(train_loader)
    wandb.log({"dataset_samples": wandb.Image(fig)})

    # 5. Modello, Loss e Optimizer
    model = CustomNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    # 6. Ciclo delle epoche
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Invia le metriche a WandB per i grafici in tempo reale
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Salva il checkpoint se migliora l'accuratezza
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Salvato nuovo miglior modello (Val Acc: {best_acc:.2f}%)")

    print(f'Training completato! Best validation accuracy: {best_acc:.2f}%')
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CustomNet on TinyImageNet')
    parser.add_argument('--data_dir', type=str, default='data/tiny-imagenet-200')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--wandb_project', type=str, default='aml-lab3')
    args = parser.parse_args()

    main(args)