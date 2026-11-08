import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("sentiment_analysis")


def setup_logger(
    name: str = "sentiment_analysis",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Khởi tạo logger đồng thời ghi ra console và file."""
    log = logging.getLogger(name)
    log.setLevel(level)

    # Tránh gắn lặp handlers nếu hàm được gọi nhiều lần
    if log.handlers:
        return log

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    return log


class WandbLogger:
    """
    Quản lý tích hợp Weights & Biases (W&B) theo đặc tả Mục 4 (Markdown.md).
    Tự động fallback an toàn nếu W&B chưa cài đặt hoặc đang chạy offline.
    """

    def __init__(
        self,
        project: str = "dl-nhom18-topic6-sentiment",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.run = None

        if not self.enabled:
            return

        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                reinit=True,
            )
            logger.info(f"🚀 W&B Run đã khởi tạo: {self.run.name} (Project: {project})")
        except Exception as e:
            logger.warning(f"⚠️ Không thể khởi tạo W&B ({e}). Chuyển sang chế độ chạy offline (không log lên web).")
            self.enabled = False
            self.wandb = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log các độ đo sau mỗi epoch hoặc bước huấn luyện."""
        if self.enabled and self.run is not None:
            self.wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Đóng phiên W&B."""
        if self.enabled and self.run is not None:
            self.wandb.finish()

