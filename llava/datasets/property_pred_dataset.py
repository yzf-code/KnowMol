import os
import random
import json
import copy
import pickle
from typing import Dict, Optional, Sequence, List
import selfies
from itertools import chain
import torch
from torch.utils.data import Dataset
import transformers
from .preprocess import preprocess, preprocess_multimodal
from .smiles2graph import smiles2graph

class PropertyPredSupervisedGraphDataset(Dataset):
    """We use MolInstruction https://huggingface.co/datasets/zjunlp/Mol-Instructions/viewer/Molecule-oriented%20Instructions/ (124K) """
    add_selfies = True
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 sample_weight=1
                ):
        super(PropertyPredSupervisedGraphDataset, self).__init__()
        with open(data_path, "rb") as f:
            list_data_dict = json.load(f)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.sample_weight = sample_weight
        if 'classification' in data_path:
            self.sub_task = 'classification'
        elif 'regression' in data_path:
            self.sub_task = 'regression'
        else :
            self.sub_task = 'other'
        
    def selfies2smiles(self, selfies_str):
        try:
            smiles_str = selfies.decoder(selfies_str)
        except:
            smiles_str = None
        return smiles_str

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        if self.add_selfies:
            instruction += f" The molecule SELFIES sequence is: {raw['input']}"
        
        if self.data_args.add_task_identifier:
            if self.sub_task == 'classification':
                instruction = '[property_pred_classification] ' + instruction
            elif self.sub_task == 'regression':
                instruction = '[property_pred_regression] ' + instruction
            elif self.sub_task == 'other':
                instruction = '[property_pred] ' + instruction

        if random.random() < 0.5:
            instruction = "<image>\n" + instruction
        else:
            instruction = instruction + "\n<image>"
        # print(instruction)
        input_selfies, target = raw['input'], str(raw['output'])
        # convert input selfies to smiles for building graph
        graph = smiles2graph(self.selfies2smiles(input_selfies))
        sources = dict(
            conversations=[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": target}
            ]
        )
        fg_name2atom = raw['fg_name2atom']
        all_fg_atoms = []
        for fg in fg_name2atom.keys():
            fg_atoms = list(chain.from_iterable(fg_name2atom[fg]))
            all_fg_atoms.append(fg_atoms)
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        if graph is not None:
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(graph is not None))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # graph exist in the data
        if graph is not None:
            data_dict['graph'] = [graph]
            data_dict['fg_atoms'] = [all_fg_atoms]
        elif self.data_args.is_multimodal:
            raise ValueError("Graph does not exist in the data, but the model is multimodal")
        return data_dict