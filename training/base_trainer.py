import os
import torch
import locale
from peft import (
    get_peft_model,
    LoraConfig,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from datetime import datetime
from accelerate import Accelerator
from abc import ABC, abstractmethod

from lib.logger import Logger
from utils.load_config import Config
from utils.data_utils import DatasetLoader

locale.getpreferredencoding = lambda: "UTF-8"

logger = Logger(__name__)


class BaseTrainer(ABC):
    def __init__(self, config_path):
        self.config = Config(config_path)
    def get_trainer(self):
        if self.config.use_4bit_bnb:
            model = self.get_model(self.config.BASE_MODEL, use_4bit_bnb=True)
        else:
            model = self.get_model(self.config.BASE_MODEL)

        tokenizer = self.get_tokenizer(self.config.BASE_MODEL)

        peft_config = None
        if self.config.USE_LORA:
            model, peft_config = self.peft_model(model)

        output_dir = self._make_run_dir()

        if self.config.USE_WANDB:
            training_arguments = TrainingArguments(
                output_dir=output_dir,
                learning_rate=self.config.LEARNING_RATE,
                num_train_epochs=self.config.NUM_EPOCHS,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                gradient_accumulation_steps=self.config.GRAD_ACCUMULATION_STEPS,
                gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
                optim=self.config.OPTIMIZER,
                weight_decay=self.config.WEIGHT_DECAY,
                max_grad_norm=self.config.MAX_GRAD_NORM,
                bf16=self.config.use_bf16,
                warmup_ratio=self.config.WARMUP_RATIO,
                lr_scheduler_type=self.config.LR_SCHEDULER_TYPE,
                save_strategy=self.config.SAVE_STRATERGY,
                save_steps=self.config.SAVE_STEPS,
                load_best_model_at_end=self.config.LOAD_BEST_MODEL_AT_END,
                evaluation_strategy=self.config.SAVE_STRATEGY,
                eval_steps=self.config.EVAL_STEPS,
                dataloader_pin_memory=True,
                dataloader_num_workers=4,
                logging_steps=self.config.LOGGING_STEPS,
                report_to=self.config.REPORT_TO
            )
        else:
            training_arguments = TrainingArguments(
                output_dir=ouput_dir,
                learning_rate=self.config.LEARNING_RATE,
                num_train_epochs=self.config.NUM_EPOCHS,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                gradient_accumulation_steps=self.config.GRAD_ACCUMULATION_STEPS,
                gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
                optim=self.config.OPTIMIZER,
                weight_decay=self.config.WEIGHT_DECAY,
                max_grad_norm=self.config.MAX_GRAD_NORM,
                bf16=self.config.use_bf16,
                warmup_ratio=self.config.WARMUP_RATIO,
                lr_scheduler_type=self.config.LR_SCHEDULER_TYPE,
                save_strategy=self.config.SAVE_STRATERGY,
                save_steps=self.config.SAVE_STEPS,
                load_best_model_at_end=self.config.LOAD_BEST_MODEL_AT_END,
                evaluation_strategy=self.config.SAVE_STRATEGY,
                eval_steps=self.config.EVAL_STEPS,
                dataloader_pin_memory=True,
                dataloader_num_workers=4,
                logging_steps=self.config.LOGGING_STEPS
            )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            peft_config=peft_config,

            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            dataset_text_field="text",

            args=training_arguments,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            packing=self.config.PACKING
        )

        return trainer
    def get_bnb_config():
        """
        Get the BitsAndBytesConfig

        :return: BiteAndBytesConfig
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return bnb_config

    def get_model(self, model: str, use_4bit_bnb: bool = False) -> AutoModelForCausalLM:
        """
        Get the model

        :return: AutoModelForCausalLM
        """
        if use_4bit_bnb:
            model = AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=self.get_bnb_config(),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        self._model_loading_postprocessing(model)
        self._model_loading_checks(model)
        self._set_more_layers_trainable(model)

        return model

    def get_tokenizer(self, model: str) -> AutoTokenizer:
        """
        Get the tokenizer

        :return: AutoTokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model=model,
            use_fast=True,
            trust_remote=True
        )
        self._set_padding_token(tokenizer)
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        return tokenizer

    @abstractmethod
    def peft_model(self, model):
        peft_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias=self.config.LORA_BIAS,
            task_type=self.config.LORA_TASK_TYPE,
            use_dora=self.config.USE_DORA
        )
        model = get_peft_model(model, peft_config)
        self._get_model_trainable_parameters()
        return model, peft_config

    def _model_loading_checks(self, model):
        """
        Perform checks on the model

        :param model: AutoModelForCausalLM
        """
        for n, p in model.named_parameters():
            if p.device.type == "meta":
                logger.error(f'{n} is on meta')

    def _model_loading_postprocessing(self, model):
        """
        Perform postprocessing on the model
        :param model:
        :return:
        """
        for params in model.parameters():
            params.requires_grad = False
            if params.ndim == 1:
                params.data = params.data.to(torch.float32)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        logger.info(f"Model loaded - {model}")

    def _get_model_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        non_trainable_params = 0
        all_param = 0

        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                non_trainable_params += param.numel()
        logger.info(
            f"Trainable params: {trainable_params} || All params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _set_padding_token(self, tokenizer):
        if '<pad>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<pad>'
        elif '<|pad|>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<|pad|>'
        elif '<unk>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<unk>'
        else:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

    def _set_more_layers_trainable(self, model):
        """
        Set more layers to be trainable

        :param model: AutoModelForCausalLM
        """
        trainable_layers = ['embed_tokens', 'input_layernorm', 'post_attention_layernorm']
        for n, p in model.named_parameters():
            if any(k in n for k in trainable_layers):
                p.requires_grad_(True)
        logger.info(f"More layers set to trainable - {trainable_layers}")

    def _make_run_dir(self):
        """
        Using DateTime to create a directory to store the run logs
        :return:
        """
        dirname = f"checkpoints/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        os.makedirs(dirname)
        return dirname

dataset_loader = DatasetLoader(DATASET_NAME)
train_dataset, test_dataset = dataset_loader.get_dataset()

# Training arguments
OUTPUT_DIR = config["OUTPUT_DIR"]
LEARNING_RATE = config["LEARNING_RATE"]

NUM_EPOCHS = config["NUM_EPOCHS"]
BATCH_SIZE = config["BATCH_SIZE"]
# effective backprop @ batch_size*grad_accum_steps
GRAD_ACCUMULATION_STEPS = config["GRAD_ACCUMULATION_STEPS"]
# speed down by ~20%, improves mem. efficiency
GRADIENT_CHECKPOINTING = config["GRADIENT_CHECKPOINTING"]

OPTIMIZER = config["OPTIMIZER"]
# OPTIMIZER = "AdamW"
# OPTIMIZER = "adamw_torch_fused" # use with pytorch compile
WEIGHT_DECAY = config["WEIGHT_DECAY"]
# examples include ["linear", "cosine", "constant"]
LR_SCHEDULER_TYPE = config["LR_SCHEDULER_TYPE"]
MAX_GRAD_NORM = config["MAX_GRAD_NORM"]  # clip the gradients after the value
# The lr takes 3% steps to reach stability
WARMUP_RATIO = config["WARMUP_RATIO"]

SAVE_STRATERGY = config["SAVE_STRATERGY"]
SAVE_STEPS = config["SAVE_STEPS"]
SAVE_TOTAL_LIMIT = config["SAVE_TOTAL_LIMIT"]
LOAD_BEST_MODEL_AT_END = config["LOAD_BEST_MODEL_AT_END"]

REPORT_TO = config["REPORT_TO"]
LOGGING_STEPS = config["LOGGING_STEPS"]
EVAL_STEPS = SAVE_STEPS

PACKING = config["PACKING"]
MAX_SEQ_LENGTH = config["MAX_SEQ_LENGTH"]


def calculate_steps():
    dataset_size = len(train_dataset)
    steps_per_epoch = dataset_size / (BATCH_SIZE * GRAD_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"Total number of steps: {total_steps}")


calculate_steps()

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,

    optim=OPTIMIZER,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    fp16=not use_bf16,
    bf16=use_bf16,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER_TYPE,

    # torch_compile=False,
    group_by_length=False,

    save_strategy=SAVE_STRATERGY,
    save_steps=SAVE_STEPS,
    # save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,

    evaluation_strategy=SAVE_STRATERGY,
    eval_steps=EVAL_STEPS,

    dataloader_pin_memory=True,
    dataloader_num_workers=4,

    logging_steps=LOGGING_STEPS,
    report_to=REPORT_TO,
)

# Define the Supervised-Finetuning-Trainer from huggingface
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,

    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",

    args=training_arguments,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=PACKING,
)

# Train model from scratch
trainer.train()

# Save the model
trainer.model.save_pretrained(NEW_MODEL)
