import os
import shutil
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def prepare_val_folder(data_dir):
    """Riorganizza la cartella val/ in sottocartelle per classe, se non è già stato fatto."""
    val_dir = os.path.join(data_dir, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    if os.path.exists(val_images_dir) and os.path.exists(annotations_file):
        print("Riorganizzazione della cartella val/ in corso...")
        with open(annotations_file) as f:
            for line in f:
                fn, cls, *_ = line.split('\t')
                cls_dir = os.path.join(val_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                shutil.copyfile(
                    os.path.join(val_images_dir, fn),
                    os.path.join(cls_dir, fn)
                )
        shutil.rmtree(val_images_dir)
        print("Cartella val/ sistemata!")


def get_dataloaders(data_dir='data/tiny-imagenet-200', batch_size=32, num_workers=2):
    """Prepara le trasformazioni, i Dataset e restituisce train_loader e val_loader."""
    prepare_val_folder(data_dir)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader