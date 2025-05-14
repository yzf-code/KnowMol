import os
import random
import json
import copy
import pickle
from PIL import Image
from typing import Dict, Optional, Sequence, List
from itertools import chain
import torch
from torch.utils.data import Dataset
import transformers
from .preprocess import preprocess, preprocess_multimodal

class MolDesignSupervisedGraphDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 sample_weight=1
                ):
        super(MolDesignSupervisedGraphDataset, self).__init__()
        with open(data_path, "rb") as f:
            list_data_dict = json.load(f)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.sample_weight = sample_weight
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']    
        if self.data_args.add_task_identifier:
            instruction = '[Molecule Design] ' + instruction

        if random.random() < 0.5:
            instruction = "<image>\n" + instruction
        else:
            instruction = instruction + "\n<image>"

        input_description = 'The molecule\'s description is: ' + raw['input']
        instruction += input_description

        output_selfies = raw['output']  
        sources = dict(
            conversations=[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        )

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=False)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        return data_dict