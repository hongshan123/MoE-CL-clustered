"""MTL5 与 Tencent3 分类数据集的读取、编码和 PyTorch Dataset 封装。"""

import json
import logging
import math
import os
import random
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from mlora.common import DataClass, DataClass2
from mlora.tokenizer import Tokenizer

from .common import AutoMetric, BasicMetric, CommonSenseTask


DEFAULT_DATA_ROOT = "/home/star/disk-7t/niuxiangqi/data"
DATA_ROOT = os.environ.get("MOE_CL_DATA_ROOT", DEFAULT_DATA_ROOT)
MTL5_DATA_PATH = os.environ.get(
    "MOE_CL_MTL5_DATA_PATH", os.path.join(DATA_ROOT, "mtl15")
)
TENCENT_DATA_PATH = os.environ.get(
    "MOE_CL_TENCENT_DATA_PATH", os.path.join(DATA_ROOT, "tencent3")
)


class QuestionAnswerTask(CommonSenseTask):
    """分类标签集合与 accuracy 度量的轻量任务定义。"""
    def __init__(self, labels: List[str]) -> None:
        super().__init__()
        self.labels_ = labels
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}
        self.label_dtype_ = torch.int

    def label_list(self) -> List[str]:
        return self.labels_

    def loading_metric(self) -> BasicMetric:
        return AutoMetric("accuracy")


# MTL5
class MTL5DataLoaderBase:
    """MTL5 JSON 分类数据的公共读取、截断和标签映射逻辑。"""
    def __init__(self,
                 dataset_name: str,
                 base_path: str,
                 label2id: Dict[str, int],
                 max_seq_length: int = 512):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.max_seq_length = max_seq_length
        self.label2id = label2id

    def get_data_path(self, is_train: bool, is_val: bool) -> str:
        """根据训练/验证/测试标志返回对应 JSON 文件路径。"""
        if is_train:
            return f"{self.base_path}/{self.dataset_name.lower()}/train.json"
        elif is_val:
            return f"{self.base_path}/{self.dataset_name.lower()}/dev.json"
        else:
            return f"{self.base_path}/{self.dataset_name.lower()}/test.json"

    def loading_data(self,
                    tokenizer: Tokenizer,
                    is_train: bool = True,
                    is_val: bool = False,
                    debug: bool = False
                    ) -> List[DataClass2]:
        """读取 JSON 样本，编码文本并过滤超过任务最大长度的样本。"""
        if debug: self.max_seq_length = 128

        data_path = self.get_data_path(is_train, is_val)
        logging.info(f"Preparing {'training' if is_train else 'validation' if is_val else 'test'} "
              f"data for {self.dataset_name}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ret: List[DataClass2] = []
        cnt = 0
        input_len = 0

        for item in data:
            prompt = item["sentence"]
            label = item["label"]

            tokens = tokenizer.encode(data=prompt)
            label = [self.label2id[label]]

            if len(tokens) > self.max_seq_length:
                continue

            cnt += 1
            input_len += len(tokens)
            ret.append({"tokens": tokens, "label": label})

            if debug and cnt >= 200: break
            # if cnt >= 1000: break

        logging.info(f"Average input length: {input_len/cnt if cnt > 0 else 0}, Data size: {cnt}")
        return ret


class AGNews(MTL5DataLoaderBase):
    def __init__(self):
        super().__init__(
            dataset_name="AGNews",
            base_path=MTL5_DATA_PATH,
            label2id={
                'World': 0,
                'Sports': 1,
                'Business': 2,
                'Science or Technology': 3
            },
            max_seq_length=512
        )


class Amazon(MTL5DataLoaderBase):
    def __init__(self):
        super().__init__(
            dataset_name="Amazon",
            base_path=MTL5_DATA_PATH,
            label2id={
                'very negative': 0,
                'negative': 1,
                'neutral': 2,
                'positive': 3,
                'very positive': 4
            },
            max_seq_length=512
        )


class DBpedia(MTL5DataLoaderBase):
    def __init__(self):
        super().__init__(
            dataset_name="DBpedia",
            base_path=MTL5_DATA_PATH,
            label2id={
                'Company': 0,
                'Educational Institution': 1,
                'Artist': 2,
                'Athlete': 3,
                'Office Holder': 4,
                'Mean of Transportation': 5,
                'Building': 6,
                'Natural Place': 7,
                'Village': 8,
                'Animal': 9,
                'Plant': 10,
                'Album': 11,
                'Film': 12,
                'Written Work': 13
            },
            max_seq_length=512
        )


class Yahoo(MTL5DataLoaderBase):
    def __init__(self):
        super().__init__(
            dataset_name="Yahoo",
            base_path=MTL5_DATA_PATH,
            label2id={
                'Society & Culture': 0,
                'Science & Mathematics': 1,
                'Health': 2,
                'Education & Reference': 3,
                'Computers & Internet': 4,
                'Sports': 5,
                'Business & Finance': 6,
                'Entertainment & Music': 7,
                'Family & Relationships': 8,
                'Politics & Government': 9
            },
            max_seq_length=480  # Yahoo使用较小的最大序列长度
        )


class MTL5Dataset(Dataset):
    """将 MTL5 原始样本补齐到统一长度，供 DataLoader 直接堆叠。"""
    def __init__(
        self,
        name: str,
        tokenizer: Tokenizer,
        cutoff_len: int = 4096,
        group_by_length: bool = True,
        expand_side: str = "right",
        is_train: bool = True,
        is_val: bool = False,
        debug: bool = False
    ):
        """Align with the evaluation method of MOCL, they are language modeling tasks."""
        self.is_train = is_train

        if name == "agnews":
            self.dataset = AGNews()
        elif name == "amazon":
            self.dataset = Amazon()
        elif name == "dbpedia":
            self.dataset = DBpedia()
        elif name == "yahoo":
            self.dataset = Yahoo()
        else:
            raise ValueError(f"Dataset {name} not found.")

        mtl5_data = self.dataset.loading_data(tokenizer, is_train, is_val, debug)
        
        max_train_tokens_len = 0
        for data in mtl5_data:
            max_train_tokens_len = max(max_train_tokens_len, len(data["tokens"]))
        logging.info(f"Max train tokens length: {max_train_tokens_len}/{cutoff_len}")

        # 按长度排序可减少同一批内 padding；随机模式用于消除顺序偏差。
        if group_by_length:
            mtl5_data.sort(key=lambda x: len(x["tokens"]), reverse=True)
        else:
            random.shuffle(mtl5_data)

        # 对齐到 8 的倍数，兼顾张量核效率与固定形状 DataLoader。
        seq_len = math.ceil(max_train_tokens_len / 8) * 8
        self.all_tokens_ = []
        self.all_mask_ = []
        self.all_label_ = []

        for data in mtl5_data:
            tokens = data["tokens"].copy()
            label = data["label"].copy()

            while len(tokens) < seq_len:
                if expand_side == "right":
                    tokens.append(tokenizer.pad_id_)
                elif expand_side == "left":
                    tokens.insert(0, tokenizer.pad_id_)
                else:
                    raise ValueError(f"Invalid padding side: {expand_side}")

            self.all_tokens_.append(tokens)
            self.all_mask_.append(tokenizer.mask_from(tokens))
            self.all_label_.append(label)

    def __len__(self):
        return len(self.all_tokens_)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.all_tokens_[idx])
        label = torch.tensor(self.all_label_[idx])
        mask = torch.tensor(self.all_mask_[idx])
        return {"tokens": tokens, "label": label, "mask": mask}


# Tencent3
class TencentDataLoaderBase:
    """Tencent3 TSV 分类数据的公共字段拼接、读取和编码逻辑。"""
    def __init__(self,
                 dataset_name: str,
                 base_path: str,
                 field_mapping: Dict[str, str],
                 max_seq_length: int = 256):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.field_mapping = field_mapping
        self.max_seq_length = max_seq_length

    def get_data_path(self, is_train: bool, is_val: bool) -> str:
        if is_train:
            return f"{self.base_path}/{self.dataset_name.lower()}/train.tsv"
        elif is_val:
            return f"{self.base_path}/{self.dataset_name.lower()}/val.tsv"
        else:
            return f"{self.base_path}/{self.dataset_name.lower()}/test.tsv"

    def format_prompt(self, data_point: pd.Series) -> str:
        """按字段映射组装英文键名的分类提示词。"""
        fields = []
        for key, field in self.field_mapping.items():
            if field in data_point:
                fields.append(f"{key}: {data_point[field]}")
        return ", ".join(fields) + "\nAnswer: "

    def loading_data(self,
                    tokenizer: Tokenizer,
                    is_train: bool = True,
                    is_val: bool = False,
                    debug: bool = False
                    ) -> List[DataClass]:
        if debug: self.max_seq_length = 128

        data_path = self.get_data_path(is_train, is_val)
        logging.info(f"Preparing {'training' if is_train else 'validation' if is_val else 'test'} "
              f"data for {self.dataset_name}")

        data = pd.read_csv(data_path, sep="\t")

        ret: List[DataClass] = []
        cnt = 0
        input_len = 0

        for _, data_point in data.iterrows():
            prompt = self.format_prompt(data_point)
            label = [data_point["label"]]

            tokens = tokenizer.encode(data=prompt)
            if len(tokens) > self.max_seq_length:
                continue

            input_len += len(tokens)
            cnt += 1
            ret.append({"tokens": tokens, "label": label})

            if debug and cnt >= 100: break

        logging.info(f"Average input length: {input_len/cnt if cnt > 0 else 0}, Data size: {cnt}")
        return ret


class ShiPinHao(TencentDataLoaderBase):
    def __init__(self):
        super().__init__(
            dataset_name="ShiPinHao",
            base_path=TENCENT_DATA_PATH,
            field_mapping={
                'Title': '标题',
                'ParentComment': '父评',
                'SubComment': '子评'
            }
        )


class XiaoShiJie(TencentDataLoaderBase):
    def __init__(self):
        super().__init__(
            dataset_name="XiaoShiJie",
            base_path=TENCENT_DATA_PATH,
            field_mapping={
                'Title': '主帖标题',
                'ParentComment': '父评信息',
                'SubComment': '子评信息'
            }
        )


class GongZhongPingLun(TencentDataLoaderBase):
    def __init__(self):
        super().__init__(
            dataset_name="GongZhongPingLun",
            base_path=TENCENT_DATA_PATH,
            field_mapping={
                'Title': '文章标题',
                'Content': '评论内容'
            }
        )


class TencentDataset(Dataset):
    """将 Tencent3 二分类样本补齐为固定长度 PyTorch Dataset。"""
    def __init__(
        self,
        name,
        tokenizer,
        cutoff_len=4096,
        group_by_length=False,
        expand_side="right",
        is_train=True,
        is_val=False,
        debug=False
    ):
        """所有 Tencent3 任务共享二分类输出头。"""
        if name == "shipinhao":
            self.dataset = ShiPinHao()
        elif name == "xiaoshijie":
            self.dataset = XiaoShiJie()
        elif name == "gongzhongpinglun":
            self.dataset = GongZhongPingLun()
        else:
            raise ValueError(f"Dataset {name} not found.")

        tencent_data = self.dataset.loading_data(tokenizer, is_train, is_val, debug)

        max_train_tokens_len = 0
        for data in tencent_data:
            max_train_tokens_len = max(max_train_tokens_len, len(data["tokens"]))
        logging.info(f"Max train tokens length: {max_train_tokens_len}/{cutoff_len}")

        # Sort by tokens length or random
        if group_by_length:
            tencent_data.sort(key=lambda x: len(x.tokens_), reverse=True)
        else:
            random.shuffle(tencent_data)

        seq_len = math.ceil(max_train_tokens_len / 8) * 8
        self.all_tokens_ = []
        self.all_mask_ = []
        self.all_label_ = []

        for data in tencent_data:
            tokens = data["tokens"].copy()
            label = data["label"].copy()

            while len(tokens) < seq_len:
                if expand_side == "right":
                    tokens.append(tokenizer.pad_id_)
                else:
                    tokens.insert(0, tokenizer.pad_id_)
            
            self.all_tokens_.append(tokens)
            self.all_mask_.append(tokenizer.mask_from(tokens))
            self.all_label_.append(label)

    def __len__(self):
        return len(self.all_tokens_)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.all_tokens_[idx])
        label = torch.tensor(self.all_label_[idx])
        mask = torch.tensor(self.all_mask_[idx])
        return {"tokens": tokens, "label": label, "mask": mask}
