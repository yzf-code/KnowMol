import os
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

class KnowMolDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 sample_weight=1):
        super(KnowMolDataset, self).__init__()
        with open(data_path, "rb") as f:
            list_data_dict = pickle.load(f)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.sample_weight = sample_weight

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        # if self.data_args.add_task_identifier:
        #     assert sources['conversations'][0]['from'] == 'human'
        #     instruction = sources['conversations'][0]['value']
        #     if instruction.startswith('<image>\n'):
        #         instruction = '<image>\n[caption] '+instruction.lstrip('<image>\n')
        #     else:
        #         instruction = '[caption] '+instruction
        #     sources['conversations'][0]['value'] = instruction
        # print(sources['conversations'][0]['value'])

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        assert sources[0]['conversations'][0]['from'] == 'human'
        if '<image>' in sources[0]["conversations"][0]['value']:
            graph = self.list_data_dict[i]['graph']
            fg_name2atom = self.list_data_dict[i]['fg_name2atom']
            all_fg_atoms = []
            for fg in fg_name2atom.keys():
                fg_atoms = list(chain.from_iterable(fg_name2atom[fg]))
                all_fg_atoms.append(fg_atoms)
            # if len(all_fg_atoms) ==0:
            #     all_fg_atoms.append([-1])

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            use_graph = True
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            use_graph = False
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=use_graph)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # graph exist in the data
        if use_graph:
            data_dict['graph'] = [graph]
            data_dict['fg_atoms'] = [all_fg_atoms]
        # elif self.data_args.is_multimodal:
        #     raise ValueError("Graph does not exist in the data, but the model is multimodal")
        return data_dict