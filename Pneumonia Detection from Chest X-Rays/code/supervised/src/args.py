import os
import json
import torch
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict
from transformers.file_utils import cached_property, torch_required

logger = logging.getLogger(__name__)


@dataclass
class DCNNArguments:
    """
    Arguments for DCNN
    """
    train_dir: Optional[str] = field(
        default="../../data/train", metadata={'help': 'training data name'}
    )
    test_dir: Optional[str] = field(
        default="../../data/test", metadata={'help': 'test data name'}
    )
    output_dir: Optional[str] = field(
        default='./output',
        metadata={"help": "The output folder where the model predictions and checkpoints will be written."},
    )
    image_size: Optional[int] = field(
        default=128,
        metadata={'help': 'The size of the normalized image'}
    )
    save_processed_data: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether save the pre-processed images to disk.'}
    )
    refresh_processed_data: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether refresh the saved pre-processed images in disk.'}
    )
    valid_ratio: Optional[float] = field(
        default=0.2,
        metadata={'help': 'Validation ratio'}
    )
    conv_dropout: Optional[float] = field(
        default=0.25,
        metadata={'help': 'drop out ratio'}
    )
    fc_dropout: Optional[float] = field(
        default=0.15,
        metadata={'help': 'drop out ratio'}
    )
    num_train_epochs: Optional[int] = field(
        default=100, metadata={'help': 'number of denoising model training epochs'}
    )
    num_valid_tolerance: Optional[int] = field(
        default=10, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    learning_rate: Optional[float] = field(
        default=0.001, metadata={'help': 'learning rate'}
    )
    warmup_ratio: Optional[int] = field(
        default=0.2, metadata={'help': 'ratio of warmup steps for learning rate scheduler'}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "Default as `linear`. See the documentation of "
                                            "`transformers.SchedulerType` for all possible values"},
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={'help': 'strength of weight decay'}
    )
    batch_size: Optional[int] = field(
        default=128, metadata={'help': 'denoising model training batch size'}
    )
    no_cuda: Optional[bool] = field(default=False, metadata={"help": "Disable CUDA even when it is available"})
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    use_cross_validation: Optional[bool] = field(
        default=False, metadata={"help": "whether use cross validation"}
    )
    debug_mode: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )

    def __post_init__(self):
        if not ((self.image_size & (self.image_size-1) == 0) and self.image_size != 0):
            raise ValueError('Image size must be the power of 2')

    # The following three functions are copied from transformers.training_args
    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        else:
            device = torch.device("cuda")
            self._n_gpu = 1

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self) -> "int":
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu


@dataclass
class DCNNConfig(DCNNArguments):
    """
    Deep CNN training configuration
    """

    def from_args(self, args: DCNNArguments) -> "DCNNConfig":
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: BertConfig)
        """
        logger.info(f'Setting {type(self)} from {type(args)}.')
        arg_elements = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr))
                        and not attr.startswith("__") and not attr.startswith("_")}
        logger.info(f'The following attributes will be changed: {arg_elements.keys()}')
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def from_config(self, config) -> "DCNNConfig":
        """
        update configuration results from other config

        Parameters
        ----------
        config: other configurations

        Returns
        -------
        self (BertConfig)
        """
        logger.info(f'Setting {type(self)} from {type(config)}.')
        config_elements = {attr: getattr(config, attr) for attr in dir(self) if not callable(getattr(config, attr))
                           and not attr.startswith("__") and not attr.startswith("_")
                           and getattr(self, attr) is None}
        logger.info(f'The following attributes will be changed: {config_elements.keys()}')
        for attr, value in config_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def save(self, file_dir: str) -> "DCNNConfig":
        """
        Save configuration to file

        Parameters
        ----------
        file_dir: file directory

        Returns
        -------
        self
        """
        if os.path.isdir(file_dir):
            file_dir = os.path.join(file_dir, 'bert_config.json')

        try:
            with open(file_dir, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.info(f"Cannot save config file to {file_dir}; "
                        f"encountered Error {e}")
            raise e
        return self

    def load(self, file_dir: str) -> "DCNNConfig":
        """
        Load configuration from stored file

        Parameters
        ----------
        file_dir: file directory

        Returns
        -------
        self
        """
        if os.path.isdir(file_dir):
            file_dir = os.path.join(file_dir, 'bert_config.json')

        logger.info(f'Setting {type(self)} parameters from {file_dir}.')

        with open(file_dir, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for attr, value in config.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self
