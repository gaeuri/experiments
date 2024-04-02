import time
import numpy as np
import torch

class Trainer:
    def __init__(self, classifier, train_dataloader, valid_dataloader, optimizer, scheduler, criterion, device, batch_size, epochs):
        self.classifier = classifier
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.best_mcc = 0.
        self.train_metrics = []
        self.valid_metrics = []

    def train_test(self, dataloader, num_batch, epoch, split='train'):
        _loss = []
        _accuracy = []
        _mcc = []
        total_test_targets = []
        total_test_preds = []

        for i, (points, targets) in enumerate(dataloader, 0):
            points = points.transpose(2, 1).to(self.device)
            targets = targets.squeeze().to(self.device)

            self.optimizer.zero_grad()
            preds, _, A = self.classifier(points)
            loss = self.criterion(preds, targets, A)

            if split == 'train':
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            pred_choice = torch.softmax(preds, dim=1).argmax(dim=1)
            correct = pred_choice.eq(targets.data).cpu().sum()
            accuracy = correct.item() / float(self.batch_size)
            mcc = mcc_metric(preds, targets)

            _loss.append(loss.item())
            _accuracy.append(accuracy)
            _mcc.append(mcc.item())

            if split == 'test':
                total_test_targets += targets.reshape(-1).cpu().numpy().tolist()
                total_test_preds += pred_choice.reshape(-1).cpu().numpy().tolist()

            if i % 100 == 0:
                print(f'\t [{epoch}: {i}/{num_batch}] ' \
                      + f'{split} loss: {loss.item():.4f} ' \
                      + f'accuracy: {accuracy:.4f} mcc: {mcc:.4f}')

        epoch_loss = np.mean(_loss)
        epoch_accuracy = np.mean(_accuracy)
        epoch_mcc = np.mean(_mcc)

        print(f'Epoch: {epoch} - {split} Loss: {epoch_loss:.4f} ' \
              + f'- {split} Accuracy: {epoch_accuracy:.4f} ' \
              + f'- {split} MCC: {epoch_mcc:.4f}')

        if split == 'test':
            return epoch_loss, epoch_accuracy, epoch_mcc, total_test_targets, total_test_preds
        else:
            return epoch_loss, epoch_accuracy, epoch_mcc

    def train(self):
        num_train_batch = int(np.ceil(len(self.train_dataloader.dataset) / self.batch_size))
        num_valid_batch = int(np.ceil(len(self.valid_dataloader.dataset) / self.batch_size))

        for epoch in range(1, self.epochs):
            self.classifier = self.classifier.train()
            _train_metrics = self.train_test(self.train_dataloader, num_train_batch, epoch, split='train')
            self.train_metrics.append(_train_metrics)

            time.sleep(4)

            with torch.no_grad():
                self.classifier = self.classifier.eval()
                _valid_metrics = self.train_test(self.valid_dataloader, num_valid_batch, epoch, split='valid')
                self.valid_metrics.append(_valid_metrics)

                time.sleep(4)

            if self.valid_metrics[-1][-1] >= self.best_mcc:
                self.best_mcc = self.valid_metrics[-1][-1]
                torch.save(self.classifier.state_dict(), f'trained_models/cls_focal_clr_2/cls_model_{epoch}.pth')
