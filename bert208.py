import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime
import os

# 配置日志
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class BERTClassifier(nn.Module):
    def __init__(self, model_path='bert-base-chinese', dropout_rate=0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)

        # 简单的分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(768, 2)
        )

        # 初始化分类器权重
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 给予正类更大的权重
        weights = torch.tensor([1.0, 4.0]).to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None, bvids=None):
        self.encodings = encodings
        self.labels = labels
        self.bvids = bvids

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        if self.bvids is not None:
            item['bvid'] = self.bvids[idx]
        return item


def preprocess_data(texts, labels=None, tokenizer=None, max_length=128):
    """预处理文本数据"""
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    if labels is not None:
        encodings['labels'] = torch.tensor(labels)
    return encodings


def save_model_info(config_name, current_step, val_metrics, train_loss=None,
                    learning_rate=None, batch_size=None, epochs=None, warmup_ratio=None):
    """保存训练指标和配置信息"""
    model_info = {
        'config': config_name,
        'org_output_dir': config_name,
        'step': current_step,
        'dev_acc': val_metrics['accuracy'],
        'dev_pre': val_metrics['precision'],
        'dev_recall': val_metrics['recall'],
        'dev_f1': val_metrics['f1'],
        'test_acc': val_metrics['accuracy'],
        'test_pre': val_metrics['precision'],
        'test_recall': val_metrics['recall'],
        'test_f1': val_metrics['f1'],
        'train_loss': train_loss,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'warmup_ratio': warmup_ratio
    }

    mode = 'a' if os.path.exists('model_metrics.csv') else 'w'
    header = not os.path.exists('model_metrics.csv')

    df = pd.DataFrame([model_info])
    df.to_csv('model_metrics.csv', mode=mode, header=header, index=False)


def evaluate_model(model, val_loader, device, criterion):
    """评估模型性能"""
    model.eval()
    val_loss = 0
    val_preds, val_labels, val_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = F.softmax(logits, dim=1)[:, 1]
            val_labels.extend(labels.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    val_labels = np.array(val_labels)
    val_probs = np.array(val_probs)

    # 尝试多个阈值，选择最佳的
    thresholds = [0.3, 0.5, 0.7]
    best_f1 = 0
    best_metrics = None

    for threshold in thresholds:
        preds = (val_probs > threshold).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            accuracy = accuracy_score(val_labels, preds)
            precision = precision_score(val_labels, preds, zero_division=0)
            recall = recall_score(val_labels, preds, zero_division=0)
            cm = confusion_matrix(val_labels, preds)
            best_metrics = (threshold, val_loss / len(val_loader), accuracy, precision,
                            recall, f1, roc_auc_score(val_labels, val_probs),
                            average_precision_score(val_labels, val_probs),
                            cm, val_probs, val_labels, preds)

    return best_metrics[1:]


def train_model(config_name, train_loader, val_loader, model, optimizer, scheduler,
                device, tokenizer, epochs=20, patience=8, eval_steps=100,
                learning_rate=None, batch_size=None, warmup_ratio=None):
    """训练模型"""
    model.to(device)
    best_val_f1 = 0
    epochs_no_improve = 0
    criterion = FocalLoss(alpha=0.75, gamma=2)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            current_step = epoch * len(train_loader) + step

            # 定期评估
            if (step + 1) % eval_steps == 0:
                avg_train_loss = train_loss / (step + 1)
                val_metrics = evaluate_model(model, val_loader, device, criterion)
                val_loss, val_accuracy, val_precision, val_recall, val_f1 = val_metrics[:5]

                metrics = {
                    'accuracy': val_accuracy,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1
                }

                save_model_info(
                    config_name,
                    current_step,
                    metrics,
                    train_loss=avg_train_loss,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    warmup_ratio=warmup_ratio
                )

                progress_bar.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_f1': f'{val_f1:.4f}'
                })

        # Epoch 结束后的评估
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = val_metrics[:5]

        logging.info(f'Epoch {epoch + 1}: Val F1 = {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0

            # 保存最佳模型
            save_dir = f'best_model_{config_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.makedirs(save_dir, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_val_f1,
                'config': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'warmup_ratio': warmup_ratio
                }
            }, os.path.join(save_dir, 'model.pt'))

            tokenizer.save_pretrained(save_dir)
            logging.info(f'Saved best model with F1 = {best_val_f1:.4f}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info('Early stopping triggered')
                break

    return model

def train_with_different_lr(csv_path, learning_rates = [1e-6, 2e-6, 3e-6, 5e-6],
                            epochs=20, batch_size=8, warmup_ratio=0.25, eval_steps=100):
    """使用不同学习率训练模型"""
    # 加载数据
    print("Loading data...")
    data = pd.read_csv(csv_path)
    texts = data['Transcription'].tolist()
    labels = data['propaganda'].tolist()
    bvids = data['BVID'].tolist()

    # 打印原始分布
    print("\nOriginal data distribution:")
    print(pd.Series(labels).value_counts())

    # 划分数据集
    train_texts, val_texts, train_labels, val_labels, train_bvids, val_bvids = train_test_split(
        texts, labels, bvids,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    for base_lr in learning_rates:
        print(f"\n{'=' * 50}")
        print(f"Training with base learning rate: {base_lr}")
        print(f"{'=' * 50}")

        # 构建配置名
        config_name = (f"chinese_bert_base/BERTClassifier_lr_{base_lr:.0e}_"
                       f"l2_0.1_bs_{batch_size}_nep_{epochs}_"
                       f"warmup_{warmup_ratio}_dp_0.5_shuffle_seed_42")

        # 初始化模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        model = BERTClassifier(model_path='bert-base-chinese')

        # 预处理数据
        print("Preprocessing data...")
        train_encodings = preprocess_data(train_texts, train_labels, tokenizer)

        # 应用SMOTE进行数据增强
        print("Applying SMOTE...")
        smote = SMOTE(sampling_strategy=0.25, random_state=42)
        X = train_encodings['input_ids'].numpy()
        y = np.array(train_labels)
        X_res, y_res = smote.fit_resample(X, y)

        print("\nAfter SMOTE - Training set distribution:")
        print(pd.Series(y_res).value_counts())

        # 创建增强后的数据集
        augmented_encodings = {
            'input_ids': torch.tensor(X_res),
            'attention_mask': torch.ones_like(torch.tensor(X_res))
        }

        # 处理验证集
        val_encodings = preprocess_data(val_texts, val_labels, tokenizer)

        # 创建数据加载器
        train_dataset = TextDataset(augmented_encodings, y_res)
        val_dataset = TextDataset(val_encodings, val_labels, val_bvids)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

        # 初始化优化器
        optimizer = AdamW([
            {'params': model.bert.parameters(), 'lr': base_lr},
            {'params': model.classifier.parameters(), 'lr': base_lr * 5}
        ], weight_decay=0.1)

        # 设置学习率调度器
        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # 训练模型
        try:
            model = train_model(
                config_name,
                train_loader,
                val_loader,
                model,
                optimizer,
                scheduler,
                device,
                tokenizer,
                epochs=epochs,
                patience=8,
                eval_steps=eval_steps,
                learning_rate=base_lr,
                batch_size=batch_size,
                warmup_ratio=warmup_ratio
            )
            logging.info(f"Successfully completed training for learning rate {base_lr}")
        except Exception as e:
            logging.error(f"Error during training with learning rate {base_lr}: {str(e)}")
            continue

def main():
    # 设置基本参数
    csv_path = '/home/carl_zhou/BERT/data/carltesting.csv'
    learning_rates = [5e-7, 3e-7, 1e-6, 5e-6, 1e-5]
    training_params = {
        'epochs': 20,
        'batch_size': 8,
        'warmup_ratio': 0.25,
        'eval_steps': 100
    }

    try:
        # 记录训练开始时间和参数
        start_time = datetime.now()
        logging.info(f"Starting training at {start_time}")
        logging.info(f"Training parameters: {training_params}")

        # 开始训练
        train_with_different_lr(
            csv_path=csv_path,
            learning_rates=learning_rates,
            **training_params
        )

        # 记录训练结束时间
        end_time = datetime.now()
        training_duration = end_time - start_time
        logging.info(f"Training completed at {end_time}")
        logging.info(f"Total training duration: {training_duration}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()