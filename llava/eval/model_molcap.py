import argparse
import torch
import os
import json
import pickle
from tqdm import tqdm
import random
import shortuuid
from itertools import chain
from typing import Generator

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, MM_ENCODER_CFG
from llava.mol_utils import check_smiles_validity
from llava.datasets.smiles2graph import smiles2graph

from typing import Dict
from transformers import TextStreamer
from torch_geometric.data import Data
import selfies


MOLCAP_INSTRUCTIONS = [
    'Could you give me a brief overview of this molecule?',
    'Could you provide a description of this molecule?',
    'Describe this molecule.',
    'Please give me some details about this molecule.',
    'Provide a brief overview of this molecule.',
    'Provide a description of this molecule.',
    'What can you tell me about this molecule?'
]


def _convert_dict_to_Data(data_dict: Dict) -> Data:
    return Data(
        x=torch.asarray(data_dict['node_feat']),
        edge_attr=torch.asarray(data_dict['edge_feat']),
        edge_index=torch.asarray(data_dict['edge_index']),
    )
    

def smiles2selfies(smiles_str):
    try:
        selfies_str = selfies.encoder(smiles_str)
    except:
        selfies_str = None
    return selfies_str
    

# def iterate_test_files(
#     args, 
#     skip_first_line:bool=False,
#     convert_smiles_to_graph:bool=False,
#     batch_size:int=4,
# )->Generator:
#     with open(args.in_file, "rt") as f:
#         if skip_first_line:
#             f.readline()
#         batch = []
#         for i, line in enumerate(f.readlines()):
#             line = line.rstrip("\n").split("\t")
#             cid, smi, gt = line
#             instruction = random.choice(MOLCAP_INSTRUCTIONS)
#             if args.add_selfies:
#                 selfies_str = smiles2selfies(smi)
#                 if selfies_str is not None:
#                     instruction += f" The compound SELFIES sequence is: {selfies_str}."
#             if args.add_task_identifier:
#                 instruction = '[caption] ' + instruction
#             if convert_smiles_to_graph:
#                 graph = smiles2graph(smi)
#                 batch.append((cid, instruction, graph, gt))
#             else:
#                 batch.append((cid, instruction, smi, gt))
#             if len(batch) == batch_size:
#                 yield zip(*batch)
#                 batch = []
#         if len(batch) > 0:
#             yield zip(*batch)

# def _length_test_file(args, skip_first_line:bool=False)->int:
#     with open(args.in_file, "rt") as f:
#         if skip_first_line:
#             f.readline()
#         return len(f.readlines())

def iterate_test_files(
    args, 
    skip_first_line:bool=False,
    convert_smiles_to_graph:bool=False,
    batch_size:int=4,
)->Generator:
    
    # with open(args.in_file, "rt") as f:
    #     if skip_first_line:
    #         f.readline()
    #     batch = []
    #     for i, line in enumerate(f.readlines()):
    #         line = line.rstrip("\n").split("\t")
    #         cid, smi, gt = line
    #         instruction = random.choice(MOLCAP_INSTRUCTIONS)
    #         if args.add_selfies:
    #             selfies_str = smiles2selfies(smi)
    #             if selfies_str is not None:
    #                 instruction += f" The compound SELFIES sequence is: {selfies_str}."
    #         if args.add_task_identifier:
    #             instruction = '[caption] ' + instruction
    #         if convert_smiles_to_graph:
    #             graph = smiles2graph(smi)
    #             batch.append((cid, instruction, graph, gt))
    #         else:
    #             batch.append((cid, instruction, smi, gt))
    #         if len(batch) == batch_size:
    #             yield zip(*batch)
    #             batch = []
    #     if len(batch) > 0:
    #         yield zip(*batch)

    with open(args.in_file, "rb") as f:
        list_data_dict = pickle.load(f)
        batch = []
        for i in range(len(list_data_dict)):
            sources = list_data_dict[i]
            if args.add_task_identifier:
                assert sources['conversations'][0]['from'] == 'human'
                instruction = sources['conversations'][0]['value']
                if instruction.startswith('<image>\n'):
                    instruction = '<image>\n[caption] '+instruction.lstrip('<image>\n')
                else:
                    instruction = '[caption] '+instruction
                sources['conversations'][0]['value'] = instruction

            assert sources['conversations'][1]['from'] == 'gpt'
            gt = sources['conversations'][1]['value']
            
            graph = list_data_dict[i]['graph']
            fg_name2atom = list_data_dict[i]['fg_name2atom']
            all_fg_atoms = []
            for fg in fg_name2atom.keys():
                fg_atoms = list(chain.from_iterable(fg_name2atom[fg]))
                all_fg_atoms.append(fg_atoms)
            batch.append((i, instruction, graph, all_fg_atoms, gt))
            if len(batch) == batch_size:
                yield zip(*batch)
                batch = []
        if len(batch) > 0:
            yield zip(*batch)
            


def _length_test_file(args, skip_first_line:bool=False)->int:
    # with open(args.in_file, "rt") as f:
    #     if skip_first_line:
    #         f.readline()
    # return len(f.readlines())
    with open(args.in_file, "rb") as f:
        list_data_dict = pickle.load(f)
        return len(list_data_dict)
        

def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # graph encoder config
    mm_encoder_cfg = MM_ENCODER_CFG(init_checkpoint=args.graph_checkpoint_path)
    mm_encoder_cfg = mm_encoder_cfg.dict()
    
    tokenizer, model, _, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, mm_encoder_cfg=mm_encoder_cfg, add_molecule_tokens=True)

    # Sampling 
    batch_size = args.batch_size
    outs = []
    with open(answers_file, "w") as ans_file:
        for data_ids, instructions, graphs, all_fg_atoms, gts in tqdm(
            iterate_test_files(args, skip_first_line=True, convert_smiles_to_graph=True, batch_size=batch_size),
            total=_length_test_file(args, skip_first_line=True)//batch_size,
        ):  
            bs = len(data_ids)
            graph_tensors = [[_convert_dict_to_Data(graph).to(device)] for graph in graphs]
            all_fg_atoms = [[all_fg_atom] for all_fg_atom in all_fg_atoms]
            cur_prompts = []
            input_ids_batch = []
            stopping_criteria_batch = []
            for idx in range(bs):
                cur_prompt = instructions[idx]
                qs = cur_prompt
                # if model.config.mm_use_im_start_end:
                #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                # else:
                #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                
                cur_prompts.append(cur_prompt)
                input_ids_batch.append(input_ids.squeeze(0))
                stopping_criteria_batch.append(stopping_criteria)
            # pad input_ids
            input_ids_batch = torch.nn.utils.rnn.pad_sequence(
                input_ids_batch,
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids_batch,
                    graphs=graph_tensors,
                    fg_atoms = all_fg_atoms,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria_batch
                )

            outputs = []
            for i in range(bs):
                output = tokenizer.decode(output_ids[i, input_ids.shape[1]:]).strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip()
                outputs.append(output)
            
            for data_id, cur_prompt, gt, output in zip(data_ids, cur_prompts, gts, outputs):
                ans_id = shortuuid.uuid()
                tmp_answer = {"data_id": data_id,
                    "prompt": cur_prompt,
                    "text": output,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "gt": gt,
                    "metadata": {}}
                outs.append(tmp_answer)

                if args.debug:
                    print("\n", {"gt": gt, "outputs": output}, "\n")

                # json_str = json.dumps(tmp_answer)
                # ans_file.write(json_str+'\n') 
                # ans_file.flush()
        
        json.dump(outs, ans_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--in-file", type=str, default="assets/chebi-20_data/test.txt")
    parser.add_argument("--answers-file", type=str, default="eval_result/answer.jsonl")
    parser.add_argument("--graph-checkpoint-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--add_selfies", action="store_true")
    parser.add_argument("--add_task_identifier", action="store_true")
    args = parser.parse_args()
    main(args)


"""
python -m llava.eval.model_molcap \
    --model-path checkpoints/llava-vicuna-v1-3-7b-finetune_lora \
    --in-file assets/chebi-20_data/test.txt \
    --answers-file eval_result/chebi20-molcap-lora-10ep.jsonl \
    --graph-checkpoint-path checkpoints/graphmvp.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --debug 
"""