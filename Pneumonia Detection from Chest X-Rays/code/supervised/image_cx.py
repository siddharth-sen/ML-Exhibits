import logging
import os
import sys
import gc
import torch
from datetime import datetime

from src.args import DCNNArguments, DCNNConfig
from src.data import ImageCxDataset
from src.train import ImageCxTrainer
from seqlbtoolkit.io import set_logging, logging_args

from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


def image_cx(args: DCNNArguments):
    set_logging(log_dir=args.log_dir)

    logging_args(args)
    set_seed(args.seed)
    config = DCNNConfig().from_args(args)

    training_dataset = test_dataset = None
    if args.train_dir:
        logger.info('Loading training dataset...')
        training_dataset = ImageCxDataset().load_file(
            file_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), args.train_dir),
            config=config
        )
        logger.info(f'Training dataset loaded, length={len(training_dataset)}')
    if not config.use_cross_validation:
        validation_dataset = training_dataset.pop_random(ratio=config.valid_ratio)
    else:
        validation_dataset = None
    if args.test_dir:
        test_dataset = ImageCxDataset().load_file(
            file_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), args.test_dir),
            config=config
        )
        logger.info(f'Test dataset loaded, length={len(test_dataset)}')

    # create output dir if it does not exist
    config.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_dir)
    if not os.path.isdir(config.output_dir):
        os.makedirs(os.path.abspath(config.output_dir))

    trainer = ImageCxTrainer(
        config=config,
        training_dataset=training_dataset,
        valid_dataset=validation_dataset,
        test_dataset=test_dataset
    ).initialize_trainer()

    if args.train_dir:
        logger.info("Start training model.")
        valid_results = trainer.train()
    else:
        trainer.load(config.output_dir, load_optimizer_and_scheduler=True)
        valid_results = None

    if args.test_dir:
        logger.info("Start testing model.")
        test_metrics = trainer.test()

        logger.info("Test results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    else:
        test_metrics = None

    # save results
    result_file = os.path.join(config.output_dir, 'dcnn-results.txt')
    logger.info(f"Writing results to {result_file}")
    with open(result_file, 'w') as f:
        if valid_results is not None:
            for i in range(len(valid_results)):
                f.write(f"[Epoch {i + 1}]\n")
                for k in ['accuracy', 'precision', 'recall', 'f1']:
                    f.write(f"  {k}: {valid_results[k][i]:.4f}")
                f.write("\n")
        if test_metrics is not None:
            f.write(f"[Test]\n")
            for k in ['accuracy', 'precision', 'recall', 'f1']:
                f.write(f"  {k}: {test_metrics[k]:.4f}")
            f.write("\n")
    structured_results_path = os.path.join(config.output_dir, 'structured-results.pt')
    torch.save(valid_results.append(test_metrics).__dict__, structured_results_path)

    logger.info("Collecting garbage.")
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Process finished!")


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(DCNNArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        chmm_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        chmm_args, = parser.parse_args_into_dataclasses()

    if chmm_args.log_dir is None:
        chmm_args.log_dir = os.path.join('logs', f'{_current_file_name}.{_time}.log')

    image_cx(args=chmm_args)
