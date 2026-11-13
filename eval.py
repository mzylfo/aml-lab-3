import argparse
import torch
from torch import nn

from models import CustomNet
from dataset import get_dataloaders

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total
    print(f'Validation Loss: {val_loss:.4f} Acc: {val_accuracy:.2f}%')
    return val_loss, val_accuracy

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Carica solo il val_loader (il train_loader non ci serve in eval)
    _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 1. Ricrea l'architettura vuota della rete
    model = CustomNet(num_classes=200).to(device)

    # 2. Carica i pesi salvati dal file checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Caricati i pesi da: {args.checkpoint}")

    # 3. Esegue la valutazione
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Risultato finale -> Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CustomNet on TinyImageNet')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='data/tiny-imagenet-200')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    main(args)