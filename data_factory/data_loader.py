# data_factory/data_loader.py

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Union, Optional

# 설정 파일 또는 클래스에서 WINDOW_GIVEN을 가져옵니다.
# 여기서는 예시로 100을 사용합니다. 필요에 따라 수정하세요.
WINDOW_GIVEN = 100

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path: str, stride: int = 1, inference: bool = False) -> None:
        """
        Time Series Dataset

        Args:
            data_path (str): 디렉토리 경로로, 모든 TimeSeries CSV 파일이 포함되어야 합니다.
            stride (int): 시퀀스 생성 시 윈도우 스트라이드.
            inference (bool): 추론 모드 여부.
        """
        self.inference = inference
        self.column_names = []
        self.file_ids = []
        self.values = []
        
        # 디렉토리 내 모든 CSV 파일 나열
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in the directory: {data_path}")
        
        # 모든 CSV 파일 읽고 병합
        df_list = []
        for csv_file in csv_files:
            csv_path = os.path.join(data_path, csv_file)
            df = pd.read_csv(csv_path)
            df_list.append(df)
        
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # '^P\d+$' 패턴에 맞는 열 이름 식별 (P1-P26)
        self.column_names = combined_df.filter(regex='^P\\d+$').columns.tolist()
        if not self.column_names:
            raise ValueError("No columns matching the pattern '^P\\d+$' found in the CSV files.")
        
        # P1-P26만 사용 (26개)
        if len(self.column_names) != 26:
            raise ValueError(f"Expected 26 P columns, but found {len(self.column_names)}.")
        
        self.file_ids = combined_df['file_id'].values if 'file_id' in combined_df.columns else None

        if self.inference:
            self.values = combined_df[self.column_names].values.astype(np.float32)
            self._prepare_inference_data()
        else:
            self._prepare_training_data(combined_df, stride)

    def _normalize_columns(self, data: np.ndarray) -> np.ndarray:
        """각 열을 독립적으로 정규화."""
        mins = data.min(axis=0, keepdims=True)
        maxs = data.max(axis=0, keepdims=True)

        # min과 max가 같은 경우, 정규화된 데이터를 0으로 설정
        is_constant = (maxs == mins)
        if np.any(is_constant):
            normalized_data = np.zeros_like(data)
            normalized_data[:, is_constant.squeeze()] = 0
            return normalized_data

        # 정규화 수행
        return (data - mins) / (maxs - mins)

    def _prepare_inference_data(self) -> None:
        """추론을 위한 데이터 준비 - 단일 시퀀스."""
        self.normalized_values = self._normalize_columns(self.values)

    def _prepare_training_data(self, df: pd.DataFrame, stride: int) -> None:
        """훈련을 위한 데이터 준비 - 윈도우 기반."""
        self.values = df[self.column_names].values.astype(np.float32)

        # 스트라이드에 따른 가능한 시작 인덱스 계산
        potential_starts = np.arange(0, len(df) - WINDOW_GIVEN, stride)

        # 윈도우 끝에서 'anomaly'가 0인 시작 인덱스 필터링
        if 'anomaly' not in df.columns:
            raise ValueError("The DataFrame must contain an 'anomaly' column for training.")
        accident_labels = df['anomaly'].values
        valid_starts = [
            idx for idx in potential_starts
            if idx + WINDOW_GIVEN < len(df) and  # 범위 확인
            accident_labels[idx + WINDOW_GIVEN] == 0  # 윈도우 끝에 이상이 없을 때
        ]
        self.start_idx = np.array(valid_starts)

        # 윈도우 추출 및 정규화
        windows = np.array([
            self.values[i:i + WINDOW_GIVEN]
            for i in self.start_idx
        ])  # Shape: [num_windows, window_size, channels]

        # 각 윈도우 정규화
        self.input_data = self._normalize_columns(windows)  # Shape: [num_windows, window_size, channels]

    def __len__(self) -> int:
        if self.inference:
            return len(self.column_names)
        return len(self.start_idx)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor, Optional[str]]]:
        if self.inference:
            col_idx = idx
            col_name = self.column_names[col_idx]
            col_data = self.normalized_values[:, col_idx]
            file_id = self.file_ids[idx] if self.file_ids is not None else None
            return {
                "column_name": col_name,
                "inputs": torch.from_numpy(col_data).unsqueeze(0)  # Shape: (1, time_steps)
            }

        window_idx = idx  # __len__은 len(start_idx)와 동일

        # 모든 채널에 대한 윈도우 데이터 추출
        window_data = self.input_data[window_idx]  # Shape: [window_size, channels]
        window_data = window_data.transpose(0, 1)  # Shape: [channels, window_size]

        return {
            "inputs": torch.from_numpy(window_data).float()  # Shape: (channels, window_size)
        }

def collate_fn_time_series(batch: List[Dict[str, Union[str, torch.Tensor, Optional[str]]]]) -> Dict[str, Union[List[str], torch.Tensor, List[Optional[str]]]]:
    """
    TimeSeriesDataset에서 반환된 딕셔너리 리스트를 배치로 변환.

    Args:
        batch (List[Dict]): TimeSeriesDataset.__getitem__에서 반환된 딕셔너리 리스트

    Returns:
        Dict: 배치 데이터
    """
    # 첫 번째 아이템에 'column_name' 키가 있는지 확인
    has_column_name = 'column_name' in batch[0]
    
    if has_column_name:
        # 추론 모드: 'column_name'과 'inputs'을 포함
        column_names = [item['column_name'] for item in batch]
        inputs = torch.stack([item['inputs'] for item in batch], dim=0)  # Shape: (batch_size, 1, time_steps)
        file_ids = [item['file_id'] for item in batch] if 'file_id' in batch[0] else None

        batch_dict = {
            'column_names': column_names,
            'inputs': inputs
        }

        if file_ids is not None:
            batch_dict['file_ids'] = file_ids

    else:
        # 훈련, 검증, 테스트 모드: 'inputs'만 포함
        inputs = torch.stack([item['inputs'] for item in batch], dim=0)  # Shape: (batch_size, channels, window_size)

        batch_dict = {
            'inputs': inputs  # Shape: [batch_size, channels, window_size]
        }

    return batch_dict

def get_loader_segment(
    data_path: str,
    batch_size: int = 64,
    win_size: int = WINDOW_GIVEN,
    stride: int = 1,
    mode: str = 'train',
    dataset: str = 'TimeSeries',
    inference: bool = False
) -> DataLoader:
    """
    TimeSeriesDataset을 위한 DataLoader 반환 함수.

    Args:
        data_path (str): 데이터 디렉토리 경로.
        batch_size (int): 배치 크기.
        win_size (int): 윈도우 크기.
        stride (int): 스트라이드.
        mode (str): 데이터 모드 ('train', 'val', 'test', 'inference').
        dataset (str): 'TimeSeries'로 고정 (다른 데이터셋은 지원하지 않음).
        inference (bool): 추론 모드 여부.

    Returns:
        DataLoader: 설정된 DataLoader.
    """
    if dataset != 'TimeSeries':
        raise ValueError(f"Unsupported dataset: {dataset}. Only 'TimeSeries' is supported.")

    # TimeSeriesDataset 초기화
    ts_dataset = TimeSeriesDataset(data_path=data_path, stride=stride, inference=inference)

    # DataLoader 생성
    data_loader = DataLoader(
        dataset=ts_dataset,
        batch_size=batch_size,
        shuffle=True if mode == 'train' else False,
        num_workers=4,  # 시스템에 따라 조정
        collate_fn=collate_fn_time_series
    )

    return data_loader
