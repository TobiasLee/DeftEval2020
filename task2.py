# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import nn
from losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy, CourageLoss
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
    DataCollatorForTokenClassification,
)
import random
# from evaluation_util import *
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    loss_type: Optional[str] = field(
        default='CrossEntropyLoss', metadata={"help": "The loss used during training."}
    )
    loss_gamma: Optional[float] = field(
        default=2, metadata={"help": "The gamma of Focal loss etc."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    train_dir: Optional[str] = field(
        default=None, metadata={"help": "A data dir of train files."}
    )
    dev_dir: Optional[str] = field(
        default=None, metadata={"help": "A data dir of dev files."}
    )

    test_dir: Optional[str] = field(default=None, metadata={"help": "A data dir of test files."})
    task_name: Optional[str] = field(default='deft_task2', metadata={"help": "A data dir of test files."})
    qa_type: bool = field(
        default=False, metadata={"help": "Whether to add a question at the beginning of the sentence."}
    )
    question_position: Optional[str] = field(
        default='suffix', metadata={"help": "Position of the question, can be chosen from ['prefix', 'suffix']."}
    )


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class TrainerWithSpecifiedLoss(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics, loss_type, loss_gamma):
        Trainer.__init__(self, model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         compute_metrics=compute_metrics)
        if loss_type == 'DiceLoss':
            self.loss_fct = DiceLoss()
        elif loss_type == 'FocalLoss':
            self.loss_fct = FocalLoss(gamma=loss_gamma)
        elif loss_type == 'LabelSmoothingCrossEntropy':
            self.loss_fct = LabelSmoothingCrossEntropy()
        elif loss_type == 'CrossEntropyLoss':
            self.loss_fct = CrossEntropyLoss()
        elif loss_type == 'CourageLoss':
            self.loss_fct = CourageLoss(gamma=loss_gamma)
        else:
            raise ValueError("Doesn't support such loss type")

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        logits = outputs[1]  # [bsz, max_token_len, class_num]
        labels = inputs['labels']  # [bsz, max_token_len]
        attention_mask = inputs['attention_mask']  # [bsz, max_token_len]
        loss = None
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, model.module.num_labels)  # [bsz * max_token_len, class_num]
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels)
                )  # [bsz * max_token_len]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(-1, model.module.num_labels), labels.view(-1))
        return loss


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_dataset('text', data_files={
        "train": [os.path.join(data_args.train_dir, k) for k in os.listdir(data_args.train_dir)][2:5],  # TODO del [:2]
        "validation": [os.path.join(data_args.dev_dir, k) for k in os.listdir(data_args.dev_dir)][2:3],
        "test": [os.path.join(data_args.test_dir, k) for k in os.listdir(data_args.test_dir)][2:3]
    })
    label_to_id = {'B-Alias-Term': 0, 'B-Alias-Term-frag': 1, 'B-Definition': 2, 'B-Definition-frag': 3,
                   'B-Ordered-Definition': 4, 'B-Ordered-Term': 5, 'B-Qualifier': 6, 'B-Referential-Definition': 7,
                   'B-Referential-Term': 8, 'B-Secondary-Definition': 9, 'B-Term': 10, 'B-Term-frag': 11,
                   'I-Alias-Term': 12, 'I-Definition': 13, 'I-Definition-frag': 14, 'I-Ordered-Definition': 15,
                   'I-Ordered-Term': 16, 'I-Qualifier': 17, 'I-Referential-Definition': 18, 'I-Referential-Term': 19,
                   'I-Secondary-Definition': 20, 'I-Term': 21, 'I-Term-frag': 22, 'O': 23}
    num_labels = len(label_to_id)
    logger.info("num_labels: %d" % num_labels)
    # Labels

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples, cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                            sep_token="[SEP]", sep_token_extra=False, pad_on_left=False, pad_token=0,
                            pad_token_segment_id=0, pad_token_label_id=-100, sequence_a_segment_id=0,
                            mask_padding_with_zero=True):
        features = {}
        # text mode examples, each line
        words = [line.strip().split("\t")[0].replace("\"", "") for line in examples['text'] if len(line) > 1]
        word_labels = [line.strip().split("\t")[4].strip() for line in examples['text'] if len(line) > 1]
        if data_args.qa_type:
            # logger.info('-' * 10 + 'cast data into qa type' + '-' * 10)
            hint_words = 'Extract name entities of definition in this sentence .'.split()  # 'I like doing NLP homework.'
            hint_word_labels = ['O' for _ in hint_words]
            if data_args.question_position == 'prefix':
                words = hint_words + words
                word_labels = hint_word_labels + word_labels
            elif data_args.question_position == 'suffix':
                words = words + hint_words
                word_labels = word_labels + hint_word_labels
            else:
                raise ValueError("do not support such question_position: %s" % data_args.question_position)
        tokens = []
        label_ids = []
        for word, label in zip(words, word_labels):
            word_tokens = tokenizer.tokenize(word)
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                # label_ids.extend([label_to_id[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                # Use the real label id for all the tokens  TODO
                label_ids.extend([label_to_id[label]] * len(word_tokens))
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        features['input_ids'] = input_ids
        features['attention_mask'] = input_mask
        features['label_ids'] = label_ids
        return features

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache,
                            remove_columns=['text'])
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets and 'test' not in datasets:
            raise ValueError("--do_eval requires a validation dataset")

        if "validation" in datasets:
            eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        else:
            eval_dataset = datasets['test']

        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.test_dir is not None:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=2)
        return {'f1': f1_score(y_true=p.label_ids, y_pred=preds),
                "precision": precision_score(y_true=p.label_ids, y_pred=preds),
                "recall": recall_score(y_true=p.label_ids, y_pred=preds),
                "accuracy": accuracy_score(y_true=p.label_ids, y_pred=preds)}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # trainer = TrainerWithSpecifiedLoss(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     compute_metrics=compute_metrics,
    #     loss_type=model_args.loss_type,
    #     loss_gamma=model_args.loss_gamma,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]

        for test_dataset, task in zip(test_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=test_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(test_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(test_dataset))

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
