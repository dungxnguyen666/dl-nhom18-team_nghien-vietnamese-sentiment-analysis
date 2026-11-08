import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên trên toàn bộ môi trường để tái lập kết quả."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Đã cố định random seed: {seed}")
