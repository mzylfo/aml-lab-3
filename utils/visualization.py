import numpy as np
import matplotlib.pyplot as plt

def denormalize(image):
    """Riporta un tensore normalizzato [C, H, W] in immagine RGB [H, W, C] tra 0 e 1."""
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def visualize_samples(dataloader, save_path=None):
    """Visualizza (o salva) una griglia 2x5 con 1 esempio per 10 classi diverse."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    classes_sampled = []
    found_classes = 0
    class_names = dataloader.dataset.classes

    for inputs, classes in dataloader:
        for img, label in zip(inputs, classes):
            label_idx = label.item()

            if label_idx not in classes_sampled:
                classes_sampled.append(label_idx)
                ax = axes[found_classes // 5, found_classes % 5]
                ax.imshow(denormalize(img))
                ax.set_title(class_names[label_idx])
                ax.axis('off')
                found_classes += 1

            if found_classes == 10:
                break
        if found_classes == 10:
            break

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
    return fig