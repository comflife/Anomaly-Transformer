import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Dict, Union

# Configuration class to manage hyperparameters
class CFG:
    WINDOW_GIVEN = 100  # Example value, adjust as needed

# Segment-based Data Loaders

class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(data_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(data_path, 'test_label.csv')).values[:, 1:]

        print("PSM - test:", self.test.shape)
        print("PSM - train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (
                torch.from_numpy(self.train[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[0:self.win_size]).float()
            )
        elif self.mode == 'val':
            return (
                torch.from_numpy(self.val[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[0:self.win_size]).float()
            )
        elif self.mode == 'test':
            return (
                torch.from_numpy(self.test[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[index:index + self.win_size]).float()
            )
        else:
            return (
                torch.from_numpy(
                    self.test[
                        (index // self.step) * self.win_size : (index // self.step) * self.win_size + self.win_size
                    ]
                ).float(),
                torch.from_numpy(
                    self.test_labels[
                        (index // self.step) * self.win_size : (index // self.step) * self.win_size + self.win_size
                    ]
                ).float()
            )


class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(data_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(data_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, "MSL_test_label.npy"))
        print("MSL - test:", self.test.shape)
        print("MSL - train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (
                torch.from_numpy(self.train[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[0:self.win_size]).float()
            )
        elif self.mode == 'val':
            return (
                torch.from_numpy(self.val[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[0:self.win_size]).float()
            )
        elif self.mode == 'test':
            return (
                torch.from_numpy(self.test[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[index:index + self.win_size]).float()
            )
        else:
            return (
                torch.from_numpy(
                    self.test[
                        (index // self.step) * self.win_size : (index // self.step) * self.win_size + self.win_size
                    ]
                ).float(),
                torch.from_numpy(
                    self.test_labels[
                        (index // self.step) * self.win_size : (index // self.step) * self.win_size + self.win_size
                    ]
                ).float()
            )


class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(data_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(data_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, "SMAP_test_label.npy"))
        print("SMAP - test:", self.test.shape)
        print("SMAP - train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (
                torch.from_numpy(self.train[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[0:self.win_size]).float()
            )
        elif self.mode == 'val':
            return (
                torch.from_numpy(self.val[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[0:self.win_size]).float()
            )
        elif self.mode == 'test':
            return (
                torch.from_numpy(self.test[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[index:index + self.win_size]).float()
            )
        else:
            return (
                torch.from_numpy(
                    self.test[
                        (index // self.step) * self.win_size : (index // self.step) * self.win_size + self.win_size
                    ]
                ).float(),
                torch.from_numpy(
                    self.test_labels[
                        (index // self.step) * self.win_size : (index // self.step) * self.win_size + self.win_size
                    ]
                ).float()
            )


class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(data_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(data_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(data_path, "SMD_test_label.npy"))
        print("SMD - test:", self.test.shape)
        print("SMD - train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (
                torch.from_numpy(self.train[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[0:self.win_size]).float()
            )
        elif self.mode == 'val':
            return (
                torch.from_numpy(self.val[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[0:self.win_size]).float()
            )
        elif self.mode == 'test':
            return (
                torch.from_numpy(self.test[index:index + self.win_size]).float(),
                torch.from_numpy(self.test_labels[index:index + self.win_size]).float()
            )
        else:
            return (
                torch.from_numpy(
                    self.test[
                        (index // self.step) * self.win_size : (index // self.step) * self.win_size + self.win_size
                    ]
                ).float(),
                torch.from_numpy(
                    self.test_labels[
                        (index // self.step) * self.win_size : (index // self.step) * self.win_size + self.win_size
                    ]
                ).float()
            )

# TimeSeriesDataset Class

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stride: int = 1, inference: bool = False) -> None:
        """
        Args:
            df: 입력 데이터프레임
            stride: 윈도우 스트라이드
            inference: 추론 모드 여부
        """
        self.inference = inference
        self.column_names = df.filter(regex='^P\\d+$').columns.tolist()
        self.file_ids = df['file_id'].values if 'file_id' in df.columns else None

        if inference:
            self.values = df[self.column_names].values.astype(np.float32)
            self._prepare_inference_data()
        else:
            self._prepare_training_data(df, stride)

    def _normalize_columns(self, data: np.ndarray) -> np.ndarray:
        """벡터화된 열 정규화"""
        mins = data.min(axis=0, keepdims=True)
        maxs = data.max(axis=0, keepdims=True)

        # mins와 maxs가 같으면 전체를 0으로 반환
        is_constant = (maxs == mins)
        if np.any(is_constant):
            normalized_data = np.zeros_like(data)
            normalized_data[:, is_constant.squeeze()] = 0
            return normalized_data

        # 정규화 수행
        return (data - mins) / (maxs - mins)

    def _prepare_inference_data(self) -> None:
        """추론 데이터 준비 - 단일 시퀀스"""
        self.normalized_values = self._normalize_columns(self.values)

    def _prepare_training_data(self, df: pd.DataFrame, stride: int) -> None:
        """학습 데이터 준비 - 윈도우 단위"""
        self.values = df[self.column_names].values.astype(np.float32)

        # 시작 인덱스 계산 (stride 적용)
        potential_starts = np.arange(0, len(df) - CFG.WINDOW_GIVEN, stride)

        # 각 윈도우의 마지막 다음 지점(window_size + 1)이 사고가 없는(0) 경우만 필터링
        accident_labels = df['anomaly'].values
        valid_starts = [
            idx for idx in potential_starts
            if idx + CFG.WINDOW_GIVEN < len(df) and  # 범위 체크
            accident_labels[idx + CFG.WINDOW_GIVEN] == 0  # 윈도우 다음 지점 체크
        ]
        self.start_idx = np.array(valid_starts)

        # 유효한 윈도우들만 추출하여 정규화
        windows = np.array([
            self.values[i:i + CFG.WINDOW_GIVEN]
            for i in self.start_idx
        ])

        # (윈도우 수, 윈도우 크기, 특성 수)로 한번에 정규화
        self.input_data = np.stack([
            self._normalize_columns(window) for window in windows
        ])

    def __len__(self) -> int:
        if self.inference:
            return len(self.column_names)
        return len(self.start_idx) * len(self.column_names)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        if self.inference:
            col_idx = idx
            col_name = self.column_names[col_idx]
            col_data = self.normalized_values[:, col_idx]
            file_id = self.file_ids[idx] if self.file_ids is not None else None
            return {
                "column_name": col_name,
                "input": torch.from_numpy(col_data).unsqueeze(-1),  # (time_steps, 1)
                "file_id": file_id
            }

        window_idx = idx // len(self.column_names)
        col_idx = idx % len(self.column_names)

        return {
            "column_name": self.column_names[col_idx],
            "input": torch.from_numpy(self.input_data[window_idx, :, col_idx]).unsqueeze(-1)
        }

# Function to get DataLoader

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', use_timeseries=False, stride=1, inference=False):
    """
    Args:
        data_path: Path to the data directory
        batch_size: Batch size for DataLoader
        win_size: Window size for segment-based loaders or WINDOW_GIVEN for TimeSeriesDataset
        step: Step size for segment-based loaders
        mode: Mode of the dataset ('train', 'val', 'test', etc.)
        dataset: Which dataset to use ('SMD', 'MSL', 'SMAP', 'PSM', 'TimeSeries')
        use_timeseries: Whether to use TimeSeriesDataset
        stride: Stride for TimeSeriesDataset
        inference: Inference mode for TimeSeriesDataset
    """
    if use_timeseries:
        # Assuming that the data_path contains a CSV file for TimeSeriesDataset
        df_path = os.path.join(data_path, 'timeseries_data.csv')  # Modify as needed
        df = pd.read_csv(df_path)
        dataset = TimeSeriesDataset(df=df, stride=stride, inference=inference)
    else:
        if dataset == 'SMD':
            dataset = SMDSegLoader(data_path, win_size, step, mode)
        elif dataset == 'MSL':
            dataset = MSLSegLoader(data_path, win_size, step, mode)
        elif dataset == 'SMAP':
            dataset = SMAPSegLoader(data_path, win_size, step, mode)
        elif dataset == 'PSM':
            dataset = PSMSegLoader(data_path, win_size, step, mode)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
