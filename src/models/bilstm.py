import collections
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.cleaner import VietnameseTextCleaner
from src.utils.metrics import compute_metrics, export_error_analysis_csv, print_metrics_summary


class Vocabulary:
    """
    Xây dựng bộ từ điển Vocabulary CHỈ từ tập Train.
    Quy tắc kỹ thuật (Mục 3.B Markdown.md):
    - 0: <pad> (dùng cho padding)
    - 1: <unk> (dùng cho từ chưa xuất hiện ngoài tập Train)
    """

    def __init__(self, min_freq: int = 1, max_size: int = 15000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_idx = 0
        self.unk_idx = 1

        self.word2idx: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word: Dict[int, str] = {0: self.pad_token, 1: self.unk_token}

    def build_vocab(self, tokenized_texts: List[List[str]]) -> "Vocabulary":
        counter = collections.Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        # Lọc theo min_freq và sắp xếp theo tần suất giảm dần
        sorted_words = [w for w, c in counter.most_common() if c >= self.min_freq]
        if self.max_size is not None:
            sorted_words = sorted_words[: self.max_size - 2]

        for idx, word in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        return self

    def numericalize(self, tokens: List[str], max_len: int = 128) -> Tuple[List[int], int]:
        """Biểu diễn câu dưới dạng chuỗi số nguyên 1D có padding/truncation."""
        ids = [self.word2idx.get(w, self.unk_idx) for w in tokens[:max_len]]
        length = max(1, len(ids))
        if len(ids) < max_len:
            ids += [self.pad_idx] * (max_len - len(ids))
        return ids, length

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)

    def load(self, filepath: str) -> "Vocabulary":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = data["idx2word"]
        return self


class SentimentDataset(Dataset):
    """Dataset quản lý chuỗi số 2D (batch_size, max_len) kiểu torch.long."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[Union[List[int], np.ndarray]] = None,
        vocab: Optional[Vocabulary] = None,
        max_len: int = 128,
        cleaner: Optional[VietnameseTextCleaner] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.cleaner = cleaner or VietnameseTextCleaner()

        # Tiền xử lý và tách token
        self.clean_texts = [self.cleaner.clean(t) for t in self.texts]
        self.tokenized_texts = [t.split() for t in self.clean_texts]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.tokenized_texts[idx]
        input_ids, length = self.vocab.numericalize(tokens, self.max_len)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class BiLSTMClassifier(nn.Module):
    """
    Nhánh 2: Mạng BiLSTM 2 chiều xử lý Masking và Padding chống nhiễu token <pad>.
    
    Quy tắc kỹ thuật (Mục 3.B Markdown.md):
    - LSTM(bidirectional=True, num_layers=2, dropout=0.3)
    - Xử lý Padding bằng pack_padded_sequence + pad_packed_sequence
    - Đầu ra: nn.Linear(hidden_dim * 2, 3)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float32),
                freeze=freeze_embeddings,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=padding_idx,
            )

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM xuất ra hidden_dim * 2 do có cả chiều thuận (forward) và nghịch (backward)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # 1. Embedding layer
        embedded = self.dropout(self.embedding(input_ids))

        # 2. Xử lý Padding bằng pack_padded_sequence để LSTM bỏ qua các token <pad>
        # lengths cần nằm trên CPU cho pack_padded_sequence
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed_embedded = pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # 3. Trích xuất hidden state cuối cùng của cả 2 chiều:
        # hidden có kích thước (num_layers * 2, batch_size, hidden_dim)
        # 2 lớp cuối cùng tương ứng với chiều thuận và nghịch của layer trên cùng
        last_forward = hidden[-2, :, :]
        last_backward = hidden[-1, :, :]
        sentence_rep = torch.cat([last_forward, last_backward], dim=1)  # (batch_size, hidden_dim * 2)

        # 4. Phân loại cảm xúc 3 lớp
        logits = self.fc(self.dropout(sentence_rep))
        return logits

