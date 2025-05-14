import os
import random
import json
import copy
import pickle
from typing import Dict, Optional, Sequence, List
from itertools import chain
import selfies
import torch
from torch.utils.data import Dataset
import transformers
from .preprocess import preprocess, preprocess_multimodal
from .smiles2graph import smiles2graph


def construct_instruct_question(product:str):
    """
    Construct instruct question for each graph
    """
    question_pools = [
        'Can you suggest some possible reagents that could have been used in the following chemical reaction?',
        'Give some possible reagents that could have been used in the following chemical reaction.',
        'Please propose potential reagents that might have been utilized in the provided chemical reaction.',
        'Please provide possible reagents based on the following chemical reaction.',
    ]
    question = random.choice(question_pools)
    question += f"\nThe product is {product}"
    return question


class ReagentPredSupervisedGraphDataset(Dataset):
    """We use MolInstruction https://huggingface.co/datasets/zjunlp/Mol-Instructions/viewer/Molecule-oriented%20Instructions/reagent_prediction (128K) """
    add_selfies = True
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 sample_weight=1
                ):
        super(ReagentPredSupervisedGraphDataset, self).__init__()
        with open(data_path, "rb") as f:
            list_data_dict = json.load(f)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.sample_weight = sample_weight
        
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
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactants = reactant.split('.')
        products = product.split('.')

        graph_list = []
        for selfie in reactants:
            reactant_smiles = self.selfies2smiles(selfie)
            graph=smiles2graph(reactant_smiles)
            graph_list.append(graph)

        for selfie in products:
            reactant_smiles = self.selfies2smiles(selfie)
            graph=smiles2graph(reactant_smiles)
            graph_list.append(graph)

        # reactant_smiles = self.selfies2smiles(reactants[0])
        # print(instruction)
        # graph=smiles2graph(reactant_smiles)


        fg_name2atom_list = self.list_data_dict[i]['fg_name2atom']
        all_fg_atoms_list = []
        for fg_name2atom in fg_name2atom_list:
            all_fg_atoms = []
            for fg in fg_name2atom.keys():
                fg_atoms = list(chain.from_iterable(fg_name2atom[fg]))
                all_fg_atoms.append(fg_atoms)
            all_fg_atoms_list.append(all_fg_atoms)

        if not self.add_selfies:
            # insert product to the instruction end
            instruction = construct_instruct_question(product)
        else:
            instruction = raw['instruction'] + f" The reaction is {input}"
        
        if self.data_args.add_task_identifier:
            instruction = '[reagent_pred] ' + instruction

        if random.random() < 0.5:
            instruction = "<image>"*len(graph_list) +"\n"+ instruction
        else:
            instruction = instruction + "\n" + "<image>"*len(graph_list)

        sources = dict(
            conversations=[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        )
        
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
            data_dict['graph'] = graph_list
            data_dict['fg_atoms'] = all_fg_atoms_list
        elif self.data_args.is_multimodal:
            raise ValueError("Graph does not exist in the data, but the model is multimodal")
        return data_dict