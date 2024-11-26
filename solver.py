# solver.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from data_factory.data_loader import get_loader_segment
from model.AnomalyTransformer import AnomalyTransformer
from utils.utils import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.val_loss2_min = float('inf')
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, f"{self.dataset}_checkpoint.pth"))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

class Solver(object):
    DEFAULTS = {
        'stride': 1,  # 기본 스트라이드 설정
        # 다른 기본값들도 여기 추가 가능
    }

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        # DataLoaders 초기화
        self.train_loader = get_loader_segment(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            stride=self.stride,
            mode='train',
            dataset='TimeSeries',
            inference=False
        )
        self.vali_loader = get_loader_segment(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            stride=self.stride,
            mode='val',
            dataset='TimeSeries',
            inference=False
        )
        self.test_loader = get_loader_segment(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            stride=self.stride,
            mode='test',
            dataset='TimeSeries',
            inference=False
        )
        self.thre_loader = get_loader_segment(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            stride=self.stride,
            mode='thre',
            dataset='TimeSeries',
            inference=False
        )

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        # enc_in=26으로 설정 (P1-P26)
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=26, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if 'inputs' in batch:
                    inputs = batch['inputs'].to(self.device)
                else:
                    raise KeyError("Batch does not contain 'inputs' key.")

                output, series, prior, _ = self.model(inputs)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # prior[u]의 차원 확인
                    # print(f"Validation: prior[{u}] shape: {prior[u].shape}")
                    # 수정된 prior_sum 계산
                    prior_sum = torch.sum(prior[u], dim=-1, keepdim=True)  # Shape: [256, 8, 100, 1]

                    normalized_prior = prior[u] / prior_sum.detach()  # Shape: [256, 8, 100, 100]

                    series_loss += (
                        torch.mean(my_kl_loss(series[u], normalized_prior)) +
                        torch.mean(my_kl_loss(normalized_prior.detach(), series[u]))
                    )
                    prior_loss += (
                        torch.mean(my_kl_loss(normalized_prior, series[u].detach())) +
                        torch.mean(my_kl_loss(series[u].detach(), normalized_prior))
                    )
                series_loss /= len(prior)
                prior_loss /= len(prior)

                rec_loss = self.criterion(output, inputs)
                loss_1.append((rec_loss - self.k * series_loss).item())
                loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name='TimeSeries')
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                if 'inputs' in batch:
                    inputs = batch['inputs'].to(self.device)
                else:
                    raise KeyError("Batch does not contain 'inputs' key.")

                self.optimizer.zero_grad()
                iter_count += 1

                output, series, prior, _ = self.model(inputs)

                # Association discrepancy 계산
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # prior[u]의 차원 확인
                    # print(f"Training: prior[{u}] shape: {prior[u].shape}")
                    # 수정된 prior_sum 계산
                    prior_sum = torch.sum(prior[u], dim=-1, keepdim=True)  # Shape: [256, 8, 100, 1]

                    normalized_prior = prior[u] / prior_sum.detach()  # Shape: [256, 8, 100, 100]

                    series_loss += (
                        torch.mean(my_kl_loss(series[u], normalized_prior)) +
                        torch.mean(my_kl_loss(normalized_prior.detach(), series[u]))
                    )
                    prior_loss += (
                        torch.mean(my_kl_loss(normalized_prior, series[u].detach())) +
                        torch.mean(my_kl_loss(series[u].detach(), normalized_prior))
                    )
                series_loss /= len(prior)
                prior_loss /= len(prior)

                rec_loss = self.criterion(output, inputs)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax 전략
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        checkpoint_path = os.path.join(str(self.model_save_path), "TimeSeries_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')  # 'reduce'는 deprecated

        # (1) 훈련 세트에서 통계
        attens_energy = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if 'inputs' in batch:
                    inputs = batch['inputs'].to(self.device)
                else:
                    raise KeyError("Batch does not contain 'inputs' key.")

                output, series, prior, _ = self.model(inputs)
                loss = torch.mean(criterion(inputs, output), dim=-1)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # prior[u]의 차원 확인
                    # print(f"Test (Train Loader): prior[{u}] shape: {prior[u].shape}")
                    # 수정된 prior_sum 계산
                    prior_sum = torch.sum(prior[u], dim=-1, keepdim=True)  # Shape: [256, 8, 100, 1]

                    normalized_prior = prior[u] / prior_sum.detach()  # Shape: [256, 8, 100, 100]

                    series_loss += (
                        my_kl_loss(series[u], normalized_prior) * temperature +
                        my_kl_loss(normalized_prior.detach(), series[u]) * temperature
                    )
                    prior_loss += (
                        my_kl_loss(normalized_prior, series[u].detach()) * temperature +
                        my_kl_loss(series[u].detach(), normalized_prior) * temperature
                    )
                series_loss /= len(prior)
                prior_loss /= len(prior)

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) 임계값 찾기
        attens_energy = []
        with torch.no_grad():
            for i, batch in enumerate(self.thre_loader):
                if 'inputs' in batch:
                    inputs = batch['inputs'].to(self.device)
                else:
                    raise KeyError("Batch does not contain 'inputs' key.")

                output, series, prior, _ = self.model(inputs)
                loss = torch.mean(criterion(inputs, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # prior[u]의 차원 확인
                    # print(f"Test (Thre Loader): prior[{u}] shape: {prior[u].shape}")
                    # 수정된 prior_sum 계산
                    prior_sum = torch.sum(prior[u], dim=-1, keepdim=True)  # Shape: [256, 8, 100, 1]

                    normalized_prior = prior[u] / prior_sum.detach()  # Shape: [256, 8, 100, 100]

                    series_loss += (
                        my_kl_loss(series[u], normalized_prior) * temperature +
                        my_kl_loss(normalized_prior.detach(), series[u]) * temperature
                    )
                    prior_loss += (
                        my_kl_loss(normalized_prior, series[u].detach()) * temperature +
                        my_kl_loss(series[u].detach(), normalized_prior) * temperature
                    )
                series_loss /= len(prior)
                prior_loss /= len(prior)

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) 테스트 세트에서 평가
        test_labels = []
        attens_energy = []
        with torch.no_grad():
            for i, batch in enumerate(self.thre_loader):
                if 'inputs' in batch:
                    inputs = batch['inputs'].to(self.device)
                else:
                    raise KeyError("Batch does not contain 'inputs' key.")

                output, series, prior, _ = self.model(inputs)
                loss = torch.mean(criterion(inputs, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # prior[u]의 차원 확인
                    # print(f"Test (Thre Loader): prior[{u}] shape: {prior[u].shape}")
                    # 수정된 prior_sum 계산
                    prior_sum = torch.sum(prior[u], dim=-1, keepdim=True)  # Shape: [256, 8, 100, 1]

                    normalized_prior = prior[u] / prior_sum.detach()  # Shape: [256, 8, 100, 100]

                    series_loss += (
                        my_kl_loss(series[u], normalized_prior) * temperature +
                        my_kl_loss(normalized_prior.detach(), series[u]) * temperature
                    )
                    prior_loss += (
                        my_kl_loss(normalized_prior, series[u].detach()) * temperature +
                        my_kl_loss(series[u].detach(), normalized_prior) * temperature
                    )
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                if 'labels' in batch:
                    test_labels.append(batch['labels'])

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        if 'labels' in batch:
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            gt = test_labels.astype(int)
            pred = (test_energy > thresh).astype(int)

            print("pred:   ", pred.shape)
            print("gt:     ", gt.shape)

            # 이상 감지 조정
            anomaly_state = False
            for i in range(len(gt)):
                if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    pred[i] = 1

            pred = np.array(pred)
            gt = np.array(gt)
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
            print(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                    accuracy, precision, recall, f_score))

            return accuracy, precision, recall, f_score
        else:
            # TimeSeriesDataset의 경우, 레이블이 없으므로 평가 메트릭을 구현해야 합니다.
            print("Evaluation metrics for TimeSeriesDataset are not implemented.")
            return None, None, None, None
