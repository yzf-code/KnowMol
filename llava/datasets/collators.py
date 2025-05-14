from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List

import torch 
from torch_geometric.data import Batch, Data
import transformers

from llava.constants import IGNORE_INDEX

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch
    

@dataclass
class GraphDataCollatorForSupervisedDataset(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph' in instances[0]:
            # g = Batch.from_data_list([self._convert_dict_to_Data(instance["graph"]) for instance in instances])
            # g = [self._convert_dict_to_Data(instance["graph"]) for instance in instances]
            g = [self._convert_list_of_dict_to_Data(instance["graph"]) for instance in instances]
            batch['graphs'] = g
        if 'fg_atoms' in instances[0]:
            # batch['fg_atoms'] = self.collate_fg_atoms([instance["fg_atoms"] for instance in instances])
            batch['fg_atoms'] = [instance["fg_atoms"] for instance in instances]
                
        return batch
    
    def _convert_dict_to_Data(self, data_dict: Dict) -> Data:
        return Data(
            x=torch.asarray(data_dict['node_feat']),
            edge_attr=torch.asarray(data_dict['edge_feat']),
            edge_index=torch.asarray(data_dict['edge_index']),
        )
    
    def _convert_list_of_dict_to_Data(self, list_of_data_dict: List) -> Data:
        processed_list = []
        for data_dict in list_of_data_dict:
            processed_list.append(Data(
                x=torch.asarray(data_dict['node_feat']),
                edge_attr=torch.asarray(data_dict['edge_feat']),
                edge_index=torch.asarray(data_dict['edge_index']),
            ))
        return processed_list

    
    # def collate_fg_atoms(self,fg_atoms_list):
    #     # 找到batch中每个样本最大list_A的长度
    #     max_fg_length = max(len(sample) for sample in fg_atoms_list)
    
    #     # 找到batch中每个样本最大子list的长度
    #     max_fgAtoms_length = max(max(len(sublist) for sublist in sample) for sample in fg_atoms_list)
        
    #     # 对每个样本进行处理
    #     processed_batch = []
    #     for sample in fg_atoms_list:
    #         # 填充每个子list到最大长度
    #         padded_sample = []
    #         for sublist in sample:
    #             # 填充子list
    #             padded_sublist = sublist + [-1] * (max_fgAtoms_length - len(sublist))
    #             padded_sample.append(padded_sublist)

    #         # 填充整个list_A到最大长度
    #         while len(padded_sample) < max_fg_length:
    #             padded_sample.append([-1] * max_fgAtoms_length)  # 用零填充空的子list

    #         processed_batch.append(padded_sample)
        
    #     # 将处理后的batch转换为张量
    #     padded_batch = torch.tensor(processed_batch)
    #     return padded_batch