from flax.training import checkpoints, train_state

from pathlib import Path


class CheckpointManager:
    def __init__(self) -> None:
        self.async_manager = checkpoints.AsyncManager()

    def save_checkpoint(self, ckpt, epoch: int, path: Path) -> None:
        checkpoints.save_checkpoint(
            ckpt_dir=path.resolve(),
            target=ckpt,
            step=epoch,
            overwrite=True,
            keep=2,
            async_manager=self.async_manager,
        )