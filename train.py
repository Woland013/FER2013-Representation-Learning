import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import FER2013Dataset, get_transforms
from models import EvolvingFERModel

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def evaluate_model(model, val_loader, device):
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds, target_names=EMOTION_LABELS)
    print("\n--- Classification Report ---")
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)
    
    return all_preds, all_labels

def run_experiment(model_type='baseline', epochs=10, batch_size=128, lr=0.001):
    wandb.init(
        project="FER2013-Representation-Learning",
        name=f"run_{model_type}_lr{lr}_bs{batch_size}",
        group=f"stage_{model_type}",
        config={
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam"
        }
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Launching Experiment: {model_type} | LR: {lr} | BS: {batch_size} on {device}")
    
    from dataset import get_train_val_loaders
    train_loader, val_loader = get_train_val_loaders('data/train.csv', batch_size=batch_size)
    
    stage_mapping = {
        'baseline': 'stage2_baseline',
        'overfit': 'stage3_deep_unregularized',
        'regularized': 'stage4_regularized'
    }
    
    actual_stage = stage_mapping.get(model_type, 'stage1_tiny')
    
    model = EvolvingFERModel(stage=actual_stage).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            for name, p in model.named_parameters():
                if p.grad is not None:
                    wandb.log({f"grad_norm/{name}": p.grad.data.norm(2).item()})
                    
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
        })
    
    print("\n--- Final Evaluation ---")
    evaluate_model(model, val_loader, device)
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{model_type}_lr{lr}_bs{batch_size}.pth")
    print(f"Model saved to checkpoints/{model_type}_lr{lr}_bs{batch_size}.pth")
    
    wandb.finish()

if __name__ == "__main__":
    run_experiment(model_type='baseline', epochs=10)
    run_experiment(model_type='overfit', epochs=20)
    run_experiment(model_type='regularized', epochs=10)