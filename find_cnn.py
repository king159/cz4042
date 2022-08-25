import copy

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from load_data import create_dataloader, create_optimizer, load_hyperparameters


class CNN(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3, units=256, dropout=0.25, stride=2):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=6000, embedding_dim=16)
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(out_channels, units)
        self.linear2 = nn.Linear(units, 1)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.dropout(x)
        x = F.relu(self.conv1(x))
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


if __name__ == '__main__':
    device = torch.device("cuda")
    criterion = nn.BCELoss()
    hypers = load_hyperparameters('cnn')
    best_acc_test = 0
    best_hyper = None
    for hy in tqdm(hypers):
        dataloaders = create_dataloader(mode='cnn', batch_size=hy['batch_size'], csv_file='IMDB_cleaned.csv')
        dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val', 'test']}
        model = CNN(out_channels=hy['filters'], kernel_size=hy['kernel_size'], units=hy["units"],
                    dropout=hy['dropout'], stride=hy['stride']).to(device)
        optimizer = create_optimizer(model=model, name=hy['optimizer'], lr=hy['lr'])
        best_acc = 0
        best_wts = None
        for epoch in range(20):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                for batch in dataloaders[phase]:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).squeeze()
                        preds = outputs.detach().cpu()
                        preds[preds >= 0.5] = 1
                        preds[preds < 0.5] = 0
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                # record parameters of the best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_wts)
        model.eval()
        running_corrects_test = 0
        for batch in dataloaders['test']:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            running_corrects_test += torch.sum(preds == labels.data)
        if running_corrects_test > best_acc_test:
            best_acc_test = running_corrects_test
            best_hyper = hy
    print(best_hyper)
    print(best_acc_test)
