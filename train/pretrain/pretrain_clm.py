#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from itertools import chain
import deepspeed
from typing import Optional, List

import datasets
import pandas as pd
import evaluate
import torch
from datasets import load_dataset
from datasets.combine import interleave_datasets
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import interleave_datasets

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    这个类是关于模型、配置和分词器的参数的，这些参数用于微调或从头开始训练模型
    """

    model_name_or_path: Optional[str] = field(  # 用于指定模型的名称或路径
        default=None,  # 设置为None，则意味着打算从头开始训练模型
        metadata={
            "help": (
                "用于权重初始化的模型的检查点。如果你想从头开始训练一个模型，请不要设置它。"
            )
        },
    )
    model_type: Optional[str] = field(  # 用于指定模型的类型。只有在从头开始训练模型时才需要
        default=None,
        metadata={"help": "如果从头开始训练模型，应该传递一个模型类型，这些模型类型来自于 MODEL_TYPES 列表: " + ", ".join(
            MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(  # 用于覆盖训练模型时的默认配置设置
        default=None,
        metadata={
            "help": (
                "当从头开始训练模型时，可以使用 config_overrides 属性来覆盖某些默认的配置设置: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(  # 用于指定预训练配置的名称或路径
        default=None, metadata={
            "help": "如果预训练配置的名称或路径与模型名称 (model_name_or_path) 不同，那么可以使用 config_name 来指定这个预训练配置"}
        # 这意味着，如果你使用的是某个特定的预训练模型，但希望使用与该模型不同的配置文件，你可以通过 config_name 来指定这个不同的配置文件的名称或路径。
    )
    tokenizer_name: Optional[str] = field(  # 用于指定预训练分词器的名称或路径
        default=None, metadata={
            "help": "如果使用的是预训练模型，但希望使用与该模型不同的分词器，那么可以通过 tokenizer_name 来指定这个不同的分词器的名称或路径。"}
    )
    cache_dir: Optional[str] = field(  # 指定从huggingface.co下载的预训练模型存储的位置
        default=None,
        metadata={"help": "这个属性是用来指定从 huggingface.co 下载的预训练模型的存储位置。"},
    )
    use_fast_tokenizer: bool = field(  # 指定是否使用快速分词器
        default=True,
        metadata={"help": "这个属性决定是否使用所谓的“快速分词器”（fast tokenizer）"},
    )  # 快速分词器通常是由 tokenizers 库支持的，它们被设计为比传统的基于Python的分词器更快、更高效。
    model_revision: str = field(  # 指定使用的特定模型版本
        default="main",
        metadata={"help": "该属性用于指定使用特定的模型版本。"},
        # 这允许用户选择不同的模型版本，例如，他们可能想使用一个特定的实验分支上的模型，或者一个特定的发布版本。
    )
    use_auth_token: bool = field(  # 是否使用在运行 huggingface-cli login 时生成的令牌（在使用私有模型时必需）
        default=False,
        metadata={
            "help": (
                "该属性允许脚本在需要时使用用户的身份验证令牌来访问私有资源或执行需要认证的操作"
            )
        },
    )
    torch_dtype: Optional[str] = field(  # 用于覆盖默认的 torch.dtype 并以此数据类型加载模型。
        default=None,
        metadata={
            "help": (
                " 属性允许用户覆盖默认的 PyTorch 数据类型 (torch.dtype)，并在此数据类型下加载模型。如果传递了 'auto' 作为值，那么数据类型将根据模型的权重自动确定。"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],  # 为用户提供了明确的选项，以便根据他们的需求和硬件能力选择合适的数据类型
        },
    )

    def __post_init__(self):  # 是一个特殊的方法 用于执行初始化后的额外逻辑。
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            # "self.config_overrides is not None"这意味着用户指定了一些要覆盖的配置设置
            # "self.config_name is not None or self.model_name_or_path is not None"这表示用户同时指定了预训练配置的名称或模型的名称/路径
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )  # 如果上述条件同时满足，方法将抛出一个 ValueError 异常。


@dataclass
class DataTrainingArguments:  # 用于封装和管理与训练和评估模型相关的数据参数
    """
    关于我们将要输入到模型中进行训练和评估的数据的参数。
    """

    dataset_name: Optional[str] = field(  # 指定要使用的数据集名称（通过 datasets 库）
        default=None, metadata={"help": "使用 dataset_name 属性来指定通过 datasets 库使用的数据集的名称。"}
        # datasets 库是一个常用于加载和处理机器学习和自然语言处理任务中数据集的库
    )
    dataset_config_name: Optional[str] = field(  # 指定要使用的数据集配置名称（通过 datasets 库）
        default=None, metadata={"help": "使用 dataset_config_name 属性来指定通过 datasets 库使用的数据集的配置名称。"}
        # 在 datasets 库中，同一个数据集可能有多个不同的配置（例如，不同的语言版本或不同的子集），dataset_config_name 允许用户指定他们想要使用的特定配置。
    )
    train_files: Optional[List[str]] = field(default=None,  # 指定输入训练数据文件（文本文件）
                                             metadata={"help": "train_files 用于指定输入训练数据的文件"})
    # 这意味着用户可以通过 train_files 属性提供一个或多个文本文件的路径，这些文件包含了用于训练模型的数据。
    validation_files: Optional[List[str]] = field(  # 指定用于评估困惑度的可选输入评估数据文件（文本文件）
        default=None,
        metadata={"help": "validation_files 用于指定用于评估的可选输入数据文件（文本文件）。"},
        # 这些文件应该是文本文件的形式，目的是在这些数据上评估模型的性能，如计算困惑度（perplexity）
    )
    max_train_samples: Optional[int] = field(  # 用于调试或加快训练，可以将训练的数量截断为此值
        default=None,
        metadata={
            "help": (
                "用于在调试或加速训练过程中限制训练样本的数量，如果设置了这个属性，[训练]样本的数量将被截断到该值。 "
                # 这对于在资源受限或需要快速测试模型性能时非常有用，因为处理较少的样本可以减少训练时间和资源消耗。
            )
        },
    )
    max_eval_samples: Optional[int] = field(  # 用于调试或加快训练，可以将评估样本的数量截断为此值
        default=None,
        metadata={
            "help": (
                "用于在调试或加速训练过程中限制评估样本的数量，如果设置了这个属性，[评估]样本的数量将被截断到该值。 "
                # 这对于在需要快速验证模型性能或进行调试时非常有用，因为处理较少的样本可以减少评估时间和资源消耗
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})  # 启用流式模式
    # 流式模式的关键特点是它允许数据被逐步读取和处理，而不是一次性加载整个数据集到内存中
    block_size: Optional[int] = field(  # 指定令牌化后的可选输入序列长度。训练数据集将被截断为此大小的块进行训练。
        default=None,
        metadata={
            "help": (
                "block_size 是一个可选的设置，用于确定令牌化后序列的长度。"
                "如果设置了 block_size，训练数据集将被截断为这个大小的块进行训练。 "
                "默认情况下，block_size 会设置为模型允许的最大输入长度，这通常是针对单个句子输入的限制，并且考虑到了特殊令牌（如开始、结束令牌）。"
                # 在自然语言处理中，原始文本首先会被分解成较小的单位，称为“令牌”（tokens）。这个过程被称为令牌化（tokenization）。
                # 例如，一个句子可以被分解成词或者更小的单位（如字节对编码）。block_size 设置的是这些令牌序列的最大长度。
                # 在模型训练时，为了保证输入数据的一致性和处理效率，通常需要将所有输入序列截断或填充到相同的长度。block_size 指定了这个固定的长度。
                # 如果一个序列的长度超过了 block_size，它将被截断；如果长度不足，则可能会被填充。
            )
        },
    )
    overwrite_cache: bool = field(  # 覆盖缓存的训练和评估集。
        default=False, metadata={"help": "当 overwrite_cache 设置为 True 时，会覆盖已缓存的训练和评估数据集。"}
        # 这意味着如果在之前的训练或评估过程中数据集已被加载并缓存，启用这个选项将强制重新加载和处理数据集，而不是使用已存在的缓存
    )
    validation_split_percentage: Optional[int] = field(  # 指定在没有验证拆分的情况下，用作验证集的训练集百分比
        default=5,
        metadata={
            "help": "在没有单独的验证数据集（validation split）的情况下，validation_split_percentage 用于指定从训练集中划分出多少百分比的数据用作验证集。"
            # 这个设置允许用户从现有的训练数据中自动划分出一部分作为验证数据
        },
    )
    preprocessing_num_workers: Optional[int] = field(  # 指定用于预处理的进程数。
        default=None,
        metadata={
            "help": "数据预处理是机器学习工作流程中的一个重要步骤，它包括清洗数据、转换数据格式、令牌化（tokenization）等操作。通过设置多个进程，可以加速这些操作的执行，特别是在处理大量数据时。"},
    )
    keep_linebreaks: bool = field(  # 指定在使用TXT文件时是否保留换行符。
        default=True, metadata={"help": "决定在处理文本文件（TXT文件）时是否保留换行符。"}
    )

    def __post_init__(self):  # 在类的初始化后自动执行
        if self.streaming:  # 如果启用了流式模式
            require_version("datasets>=2.0.0",
                            "The streaming feature requires `datasets>=2.0.0`")  # 检查 datasets 库的版本是否满足要求

        if self.dataset_name is None and self.train_files is None and self.validation_files is None:  # 如果没有指定任何一个
            raise ValueError("Need either a dataset name or a training/validation file.")  # 报错
        else:  # 如果指定了 train_files 或 validation_files，验证文件扩展名
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # ModelArguments（模型相关的参数），DataTrainingArguments（数据训练相关的参数），以及 TrainingArguments（训练过程相关的参数）。

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):  # 检查是否只传入了一个参数，并且这个参数是一个 .json 文件的路径
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        # 解析这个JSON文件以获取参数
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        # 解析后的参数分别被赋值给 model_args, data_args, training_args 这三个变量，它们分别存储了与模型相关、数据训练相关和训练过程相关的参数。

    send_example_telemetry("run_clm", model_args, data_args)  # 这行代码发送遥测数据，用于跟踪脚本的使用情况。

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        # 检查是否存在输出目录（output_dir），是否计划进行训练（do_train），并且是否没有设置覆盖输出目录（overwrite_output_dir）的标志。
        last_checkpoint = get_last_checkpoint(training_args.output_dir)  # 尝试获取最后的检查点。
        if last_checkpoint is None and len(
                os.listdir(training_args.output_dir)) > 0:  # 如果 last_checkpoint 为 None 并且输出目录不为空
            raise ValueError(  # 则抛出一个错误
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )  # 这是为了避免无意中覆盖已有的训练结果。
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:  # 如果找到了检查点且没有指定从特定检查点恢复
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )  # 则记录一条信息，说明将从检测到的检查点恢复训练。

    # Set seed before initializing model.
    set_seed(training_args.seed)  # 确保模型初始化和后续的随机过程（如数据洗牌）在不同运行之间保持一致。

    # 用户可以提供自己的 CSV/JSON/TXT 文件作为训练和评估数据，或者指定一个可在 Hugging Face 数据集 Hub 上找到的公共数据集的名称。如果选择后者，数据集将自动从 Hugging Face 数据集
    # Hub 下载。 对于 CSV 或 JSON 格式的文件，脚本默认使用名为 'text' 的列作为数据源。如果没有找到名为 'text' 的列，则默认使用文件中的第一列。这个行为可以根据需要进行调整。
    # 在分布式训练环境中，load_dataset 函数确保同一时间只有一个本地进程下载数据集。这是为了避免在多个进程尝试同时下载同一个数据集时可能发生的冲突或资源浪费。
    if True:
        data_files = {}  # 初始化一个空字典，用于存储数据集文件的路径
        dataset_args = {}  # 初始化另一个空字典，用于存储加载数据集时的附加参数

        # 检查 data_args.train_files 和 data_args.validation_files 是否非空，如果是，则将它们分别添加到 data_files 字典的 "train" 和
        # "validation" 键下。
        if data_args.train_files is not None:
            print(data_args.train_files)
            data_files["train"] = data_args.train_files
            print('训练文件总个数', len(data_args.train_files))
        if data_args.validation_files is not None:
            data_files["validation"] = data_args.validation_files

        # 计算文件扩展名，用于后续确定如何加载数据集。如果是文本文件（.txt），则将扩展名改为 "text"。
        extension = (
            data_files["train"][0].split(".")[-1]
            if data_files["train"] is not None
            else data_args.validation_files.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        raw_datasets = load_dataset(  # 根据提供的文件扩展名、文件路径和其他参数来加载数据集
            extension,
            data_files=data_files,
            streaming=data_args.streaming,
            cache_dir=os.path.join(training_args.output_dir, 'dataset_cache'),
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        if data_args.streaming:  # 如果启用了流式模式
            raw_datasets = raw_datasets.shuffle(seed=training_args.seed, buffer_size=1000000)  # 则对数据集应用随机洗牌

        # 如果没有单独的验证数据集，则将训练数据集分割成两部分：训练集和验证集。
        # 通过指定分割比例来加载训练和验证数据集
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # 有关如何加载各种标准或自定义数据集 https://huggingface.co/docs/datasets/index
    # Load pretrained model and tokenizer
    #
    # 在分布式训练中，from_pretrained 方法确保在同一时间只有一个本地进程可以并发地下载模型和词汇表。这是为了避免在多个进程尝试同时下载相同资源时可能发生的冲突或重复下载，从而提高效率和稳定性。

    config_kwargs = {  # 存储加载预训练模型配置的关键字参数
        "cache_dir": model_args.cache_dir,  # 指定了模型配置文件的缓存目录
        "revision": model_args.model_revision,  # 指定模型的特定版本或修订
        "use_auth_token": True if model_args.use_auth_token else None,  # 用于访问私有模型或需要认证的资源
    }
    if model_args.config_name:  # 如果提供了 预训练配置的名称或路径
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)  # 则使用这个名称从预训练模型库中加载配置
    elif model_args.model_name_or_path:  # 如果提供了 config_name
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)  # 则使用这个路径或名称从预训练模型库中加载配置。
    else:
        config = CONFIG_MAPPING[model_args.model_type]()  # 创建一个新的配置实例
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")  # 检查是否有任何要覆盖的配置设置
            config.update_from_string(model_args.config_overrides)  # 将指定的配置覆盖到当前配置上
            logger.info(f"New config: {config}")

    # =======================================分词器加载=======================================
    print(training_args.local_rank, '开始加载分词器')
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,  # 指定一个目录来存储下载或缓存的分词器文件
        "use_fast": model_args.use_fast_tokenizer,  # 指示是否使用快速分词器
        "revision": model_args.model_revision,  # 用于指定分词器的特定版本或修订。
        "use_auth_token": True if model_args.use_auth_token else None,  # 指定是否使用认证令牌。
    }
    if model_args.tokenizer_name:  # 检查 model_args 中是否有 tokenizer_name 参数指定。
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)  #
        # 则使用这个名称从预训练模型库中加载分词器。
    elif model_args.model_name_or_path:  # 如果提供了 model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)  #
        # 则使用这个路径或名称从预训练模型库中加载分词器
    else:
        raise ValueError(
            "脚本不支持从头开始创建新的分词器实例."
            "可以使用另一个脚本来创建并保存分词器, using --tokenizer_name."
        )
    print(training_args.local_rank, 'end load tokenizer')
    # ======================================================================================

    # =======================================模型加载========================================
    print(training_args.local_rank, 'start load model')
    if model_args.model_name_or_path:
        torch_dtype = (  # 根据 model_args.torch_dtype 设置 PyTorch 的数据类型
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(  # 从预训练模型加载因果语言模型
            model_args.model_name_or_path,  # 指定了模型的名称或路径
            from_tf=bool(".ckpt" in model_args.model_name_or_path),  # 检查是否需要从 TensorFlow 的检查点（.ckpt 文件）加载模型
            config=config,  # 传递模型的配置对象
            cache_dir=model_args.cache_dir,  # 指定缓存目录，用于存储下载的预训练模型文件。
            revision=model_args.model_revision,  # 指定模型的版本或修订
            use_auth_token=True if model_args.use_auth_token else None,  # 如果需要，提供认证令牌以访问私有模型或受限资源。
        )
    else:  # 没有提供预训练模型的名称或路径
        model = AutoModelForCausalLM.from_config(config)  # 根据提供的配置对象 config 初始化一个新的因果语言模型。这意味着模型将从零开始创建，而不是基于预训练的模型。
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())  # 计算模型的参数总数。
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")
    print(training_args.local_rank, 'end load model')
    # ======================================================================================

    # 嵌入层的大小仅在必要时被调整，以避免索引错误。这通常发生在使用的词汇量（vocab）大小与预训练模型中的嵌入层大小不匹配时。
    # 当您使用预训练模型并向其添加新的词汇（例如，特定于领域的术语），这些新词汇不在原始预训练模型的词汇表中时，就需要调整嵌入层的大小。
    # 调整嵌入层的目的是确保模型能够处理这些新词汇而不会出现索引错误。

    embedding_size = model.get_input_embeddings().weight.shape[0]  # 获取了模型输入嵌入层的大小
    if len(tokenizer) > embedding_size:  # 检查分词器的词汇量是否大于模型当前的嵌入层大小
        model.resize_token_embeddings(len(tokenizer))  # 调整模型的嵌入层大小，使其与分词器的词汇量相匹配。
    # *****************************数据预处理*********************************
    # 第一步——对文本进行令牌化
    if training_args.do_train:  # 检查是否设置了进行训练的标志(判断是训练模型还是评估模型)
        if data_args.streaming:  # 进一步检查数据是否以流式方式提供
            dataset_head = raw_datasets["train"].take(3)  # 从训练集取出前三条数据作为样本
            print(list(dataset_head))
            column_names = list(list(dataset_head)[0].keys())  # 从这些样本中提取列名
        else:
            column_names = list(raw_datasets["train"].features)  # 直接从训练集的特征中提取列名
    else:  # 如果未设置训练标志（评估模型）
        if data_args.streaming:
            dataset_head = raw_datasets["validation"].take(3)  # 同样操作验证集
            column_names = list(list(dataset_head)[0].keys())
        else:
            column_names = list(raw_datasets["validation"].features)
    print(column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]  # 确定要用于训练或验证的文本列的名称

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(['<s>' + item + '</s>' for item in examples[text_column_name]])  #
            # 对每个文本样本执行令牌化，表示序列的开始和结束
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):  # 确保在分布式训练环境中，主进程首先执行令牌化映射
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(  # 这包括批量处理、多进程处理，以及从缓存加载数据的选项。
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                batch_size=60000,
            )

    # 设置文本处理中使用的“块大小”,即处理文本时每个数据块的最大长度
    if data_args.block_size is None:  # 如果用户没有指定块大小
        block_size = tokenizer.model_max_length  # 将块大小设置为分词器支持的最大长度。
        if block_size > 1024:  # 检查分词器的最大长度是否超过了默认的块大小限制
            logger.warning(
                "发出警告，表明分词器支持的长度超过默认值，并提醒用户可以通过设置 --block_size 参数来自定义块大小。"
            )
            block_size = 1024  # 将块大小设置为 1024
    else:  # 如果用户指定了块大小
        if data_args.block_size > tokenizer.model_max_length:  # 检查用户指定的块大小是否超过了分词器的最大长度
            logger.warning(
                # 如果超过了，发出警告，表明用户指定的块大小超出了模型的最大处理长度，并自动使用分词器的最大长度作为块大小。
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)  # 最终的块大小是用户指定值和分词器最大长度之间的较小者

    # 数据处理函数，其作用是将数据集中的所有文本拼接起来，并根据指定的块大小（block_size）生成数据块。
    def group_texts(examples):  # 处理和组织文本数据，以便进行机器学习模型的训练
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}  # 拼接所有文本
        # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])  # 计算拼接后的文本总长度。

        # 如何处理长度不一致的数据,在将文本数据切分成等长的块后，可能会剩下一小部分数据,可视情况选择丢弃或者填充

        if total_length >= block_size:  # 如果总长度>=块大小
            total_length = (total_length // block_size) * block_size  # 调整总长度为块大小的整数倍，这是为了确保所有数据块的长度一致
        result = {  # 这部分代码将拼接后的长文本切分成多个小块，每个块的长度为 block_size。
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))       
        logger.info("group texts input examples length%d after_group size%d" % (
            len(examples['input_ids']), len(result["input_ids"])))
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # 进行文本数据的最终处理和准备工作
    with training_args.main_process_first(desc="grouping texts together"):  # 确保在分布式训练环境中，主进程（main process）首先执行数据处理任务
        if not data_args.streaming:  # 如果数据不是以流式方式提供的
            lm_datasets = tokenized_datasets.map(  # 将文本数据分组和组织成指定大小的块
                group_texts,
                batched=True,  # 表示数据会批量处理
                num_proc=data_args.preprocessing_num_workers,  # 设置使用多少个进程来并行处理数据，这有助于加快数据处理速度
                load_from_cache_file=not data_args.overwrite_cache,  # 控制是否从缓存中加载数据。
                desc=f"Grouping texts in chunks of {block_size}",  # 提供了描述信息，说明正在进行的操作。
                batch_size=40000,  # 设置每批处理的数据量
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                batch_size=60000,
            )

    print(training_args.local_rank, '开始选择用于训练的数据集')
    if training_args.do_train:  # 检查是否设置了训练标志
        if "train" not in tokenized_datasets:  # 检查 tokenized_datasets 中是否包含名为 "train" 的数据集
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None and data_args.streaming == False:  # 如果设置了最大训练样本数且数据不是以流式方式提供的
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)  # 计算实际使用的训练样本数，不超过用户设定的最大值
            train_dataset = train_dataset.select(range(max_train_samples))  # 从训练数据集中选择指定数量的样本
    print(training_args.local_rank, 'end select train_dataset')

    if training_args.do_eval:  # 检查是否设置了评估标志
        if "validation" not in tokenized_datasets:  # 检查处理后的数据集 (tokenized_datasets) 中是否包含名为 "validation" 的部分
            raise ValueError("--do_eval requires a validation dataset")
        print(training_args.local_rank, '选择和准备用于评估（evaluation）的数据集')
        eval_dataset = lm_datasets["validation"]  # 从处理过的数据集 (lm_datasets) 中选择名为 "validation" 的部分作为评估数据集
        if data_args.max_eval_samples is not None and data_args.streaming == False:  # 检查是否设置了最大评估样本数，且数据不是以流式方式提供的
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)  # 计算实际使用的评估样本数，不超过用户设定的最大值
            eval_dataset = eval_dataset.select(range(max_eval_samples))  # 从评估数据集中选择指定数量的样本
        print(training_args.local_rank, 'end select eval_dataset')

        def preprocess_logits_for_metrics(logits, labels):  # 处理模型产生的逻辑输出（通常称为 logits），以便于后续的性能评估。
            if isinstance(logits, tuple):
                # 模型的输出（logits）可能不仅包含了预测值，还可能包含其他额外的张量（tensors）
                logits = logits[0]
            return logits.argmax(dim=-1)

        print(training_args.local_rank, '开始加载 metric（用于评估模型性能的度量标准）')
        metric = evaluate.load("accuracy.py")
        print(training_args.local_rank, 'end load metric')

        def compute_metrics(eval_preds):  # 用于计算模型在评估过程中的性能指标。
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics, but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    print(training_args.local_rank, '开始初始化 Trainer')
    trainer = Trainer(  # 创建一个 Trainer 实例
        model=model,  # 指定要训练或评估的模型。
        args=training_args,  # 提供训练相关的参数
        train_dataset=IterableWrapper(train_dataset) if training_args.do_train else None,  # 如果设置了训练标志，将训练数据集包装成可迭代对象
        eval_dataset=IterableWrapper(eval_dataset) if training_args.do_eval else None,  # 如果设置了评估标志，将评估数据集包装成可迭代对象
        tokenizer=tokenizer,  # 指定用于文本处理的分词器
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,  # 设置数据整理器，用于将多个数据样本整理成一个批次。
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,  #
        # 如果设置了评估标志且不使用 TPU，指定计算性能指标的函数。

        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        # 如果设置了评估标志且不使用 TPU，指定预处理 logits 的函数。

        # callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
    )

    if training_args.do_train:  # 确保只有在设置了训练标志时才执行训练过程
        checkpoint = None  # 默认设置检查点为 None
        if training_args.resume_from_checkpoint is not None:  # 如果有指定从某个检查点恢复训练
            checkpoint = training_args.resume_from_checkpoint  # 则更新检查点变量
        elif last_checkpoint is not None:  # 如果没有指定检查点，但存在最后一个检查点
            checkpoint = last_checkpoint  # 则从最后一个检查点恢复训练

        print(training_args.local_rank, '开始训练')
        train_result = trainer.train(resume_from_checkpoint=checkpoint)  # 开始训练过程。如果指定了检查点，训练将从该检查点恢复
        trainer.save_model()  # 训练完成后，保存模型

        metrics = train_result.metrics  # 获取训练过程中收集的指标

        max_train_samples = (  # 计算实际使用的最大训练样本数量
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))  # 更新训练指标，加入实际使用的训练样本数

        trainer.log_metrics("train", metrics)  # 记录训练指标
        trainer.save_metrics("train", metrics)  # 保存训练指标
        trainer.save_state()  # 保存训练器的状态，包括优化器和调度器的状态

    # 进行模型评估
    if training_args.do_eval:  # 确保只有在设置了评估标志时才执行评估过程
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()  # 开始评估过程，并收集评估指标

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)  # 计算实际使用的最大评估样本数量
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))  # 在评估指标中加入实际使用的评估样本数
        try:
            perplexity = math.exp(metrics["eval_loss"])  # 尝试计算困惑度，它是评估损失（eval_loss）的指数
        except OverflowError:
            perplexity = float("inf")  # 如果计算中出现溢出错误（通常是因为损失太大），则将困惑度设置为无穷大
        metrics["perplexity"] = perplexity  # 将计算出的困惑度添加到评估指标中

        trainer.log_metrics("eval", metrics)  # 记录评估指标
        trainer.save_metrics("eval", metrics)  # 保存评估指标


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
