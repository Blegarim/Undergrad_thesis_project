import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

@torch.no_grad()
def visualize_prediction(dataset, model, idx, device='cpu', label_names=None):
    model.eval()
    model.to(device)

    images, motions, labels = dataset[idx]

    images = images.unsqueeze(0).to(device)
    motions = motions.unsqueeze(0).to(device)

    outputs = model(images, motions)

    print('\n====== PREDICTION ======')
    for head in outputs:
        logits = outputs[head].squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs).item()

        gt_label = labels[head].item() if isinstance(labels, dict) else labels.item()

        # Use label names if provided
        pred_name = label_names[head][pred_class] if label_names and head in label_names else str(pred_class)
        gt_name = label_names[head][gt_label] if label_names and head in label_names else str(gt_label)

        print(f"{head.upper()}: GT={gt_name} | Pred={pred_name} | Prob={probs[pred_class]:.2f}")
    print('========================\n')

    # Plot sequences
    seq_len = images.shape[1]
    fig, axes = plt.subplots(1, seq_len, figsize=(seq_len * 2, 3))
    if seq_len == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = F.to_pil_image(images[0, i].cpu())
        ax.imshow(img)
        ax.axis('off')

    plt.suptitle('Prediction vs Ground Truth', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.show()