import argparse
import torch
import os
import json
from tqdm import tqdm
import random
import shortuuid
from typing import Generator, Dict
import selfies
from itertools import chain
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, MM_ENCODER_CFG
from llava.mol_utils import check_smiles_validity
from llava.datasets.smiles2graph import smiles2graph

from torch_geometric.data import Data


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

# def _convert_dict_to_Data(data_dict: Dict) -> Data:
#     return Data(
#         x=torch.asarray(data_dict['node_feat']),
#         edge_attr=torch.asarray(data_dict['edge_feat']),
#         edge_index=torch.asarray(data_dict['edge_index']),
#     )
def _convert_list_of_dict_to_Data(list_of_data_dict, device) -> Data:
        processed_list = []
        for data_dict in list_of_data_dict:
            processed_list.append(Data(
                x=torch.asarray(data_dict['node_feat']),
                edge_attr=torch.asarray(data_dict['edge_feat']),
                edge_index=torch.asarray(data_dict['edge_index']),
            ).to(device))
        return processed_list   

def selfies2smiles(selfies_str):
    try:
        smiles_str = selfies.decoder(selfies_str)
    except:
        smiles_str = None
    return smiles_str

def get_fg_list(fg_name2atom):
    all_fg_atoms = []
    for fg in fg_name2atom.keys():
        fg_atoms = list(chain.from_iterable(fg_name2atom[fg]))
        all_fg_atoms.append(fg_atoms)
    return all_fg_atoms

def iterate_test_files(
    args, 
    convert_smiles_to_graph:bool=True,
    batch_size:int=4,
)->Generator:
    with open(args.in_file, "rb") as f:
        list_data_dict = json.load(f)
        
        batch = []
        for i, raw in enumerate(list_data_dict):
            if args.task == "retrosynthesis":
                graph = smiles2graph(selfies2smiles(raw['input']))
                if args.add_selfies:
                    instruction = raw['instruction'] + f" The product is: {raw['input']}"
                else:
                    instruction = raw['instruction']
                if args.add_task_identifier:
                    instruction = '[retrosynthesis] ' + instruction
                    
                all_fg_atoms = get_fg_list(raw['fg_name2atom'])
                batch.append((instruction, [graph], [all_fg_atoms], raw['output']))

            elif args.task == "reagent_pred":
                reactant, product = raw['input'].split(">>")
                reactants = reactant.split('.')
                # graph = smiles2graph(selfies2smiles(reactants[0]))
                
                products = product.split('.')

                graph_list = []
                for selfie in reactants:
                    reactant_smiles = selfies2smiles(selfie)
                    graph=smiles2graph(reactant_smiles)
                    graph_list.append(graph)

                for selfie in products:
                    reactant_smiles = selfies2smiles(selfie)
                    graph=smiles2graph(reactant_smiles)
                    graph_list.append(graph)

                # all_fg_atoms = get_fg_list(raw['fg_name2atom'])
                fg_name2atom_list = raw['fg_name2atom']
                all_fg_atoms_list = []
                for fg_name2atom in fg_name2atom_list:
                    all_fg_atoms = get_fg_list(fg_name2atom)
                    all_fg_atoms_list.append(all_fg_atoms)
                
                if not args.add_selfies:
                    # insert product to the instruction end
                    instruction = construct_instruct_question(product)
                else:
                    instruction = raw['instruction'] + f" The reaction is {raw['input']}"
                if args.add_task_identifier:
                    instruction = '[reagent_pred] ' + instruction
                    
                graph_list=[graph_list[0]]
                all_fg_atoms_list=[all_fg_atoms_list[0]]
                
                batch.append((instruction, graph_list, all_fg_atoms_list, raw['output']))

            elif args.task == "forward_pred":
                inputs = raw['input'].split('.')
                # graph = smiles2graph(selfies2smiles(inputs[0]))
                graph_list = []
                for selfie in inputs:
                    reactant_smiles = selfies2smiles(selfie)
                    graph=smiles2graph(reactant_smiles)
                    graph_list.append(graph)
                # all_fg_atoms = get_fg_list(raw['fg_name2atom'])
                fg_name2atom_list = raw['fg_name2atom']
                all_fg_atoms_list = []
                for fg_name2atom in fg_name2atom_list:
                    all_fg_atoms = get_fg_list(fg_name2atom)
                    all_fg_atoms_list.append(all_fg_atoms)

                instruction = raw['instruction']
                if args.add_selfies:
                    instruction += " " + raw['input']
                else:
                    # insert the remaining reactants to the instruction
                    if len(inputs) > 1:
                        instruction += f" The other joint reactants are: {','.join(inputs[1:])}"
                if args.add_task_identifier:
                    instruction = '[forward_pred] ' + instruction
                    
                
                batch.append((instruction, graph_list, all_fg_atoms_list, raw['output']))

            elif args.task == "property_pred_classification":
                instruction = raw['instruction']
                if args.add_selfies:
                    instruction += f" The compound SELFIES sequence is: {raw['input']}"
                input_selfies, target = raw['input'], str(raw['output'])
                graph = smiles2graph(selfies2smiles(input_selfies))
                if args.add_task_identifier:
                    instruction = '[property_pred_classification] ' + instruction
                
                all_fg_atoms = get_fg_list(raw['fg_name2atom'])
                batch.append((instruction, [graph], [all_fg_atoms], raw['output']))

            elif args.task == "property_pred_regression":
                instruction = raw['instruction']
                if args.add_selfies:
                    instruction += f" The compound SELFIES sequence is: {raw['input']}"
                input_selfies, target = raw['input'], str(raw['output'])
                graph = smiles2graph(selfies2smiles(input_selfies))
                if args.add_task_identifier:
                    instruction = '[property_pred_regression] ' + instruction

                all_fg_atoms = get_fg_list(raw['fg_name2atom'])
                batch.append((instruction, [graph], [all_fg_atoms], raw['output']))
            
            elif args.task == "molecule_design":
                instruction = raw['instruction']
                input_description = 'The molecule\'s description is: ' + raw['input']
                instruction += input_description

                if args.add_task_identifier:
                    instruction = '[Molecule Design] ' + instruction
                
                #返回None是不是ok的啊？
                batch.append((instruction, None, None, raw['output']))

            elif args.task == "demo":
                instruction = raw['instruction']
                if args.add_selfies:
                    instruction += f" The compound SELFIES sequence is: {raw['input']}"
                graph = smiles2graph(selfies2smiles(raw['input']))
                all_fg_atoms = get_fg_list(raw['fg_name2atom'])
                batch.append((instruction, [graph],[all_fg_atoms],''))

            else:
                raise NotImplementedError
            
            if len(batch) == batch_size:
                yield zip(*batch)
                batch = []
        if len(batch) > 0:
            yield zip(*batch)


def _length_test_file(args)->int:
    with open(args.in_file, "rb") as f:
        list_data_dict = json.load(f)
        return len(list_data_dict)


def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # graph encoder config
    mm_encoder_cfg = MM_ENCODER_CFG(init_checkpoint=args.graph_checkpoint_path)
    mm_encoder_cfg = mm_encoder_cfg.dict()
    print(mm_encoder_cfg)
    tokenizer, model, _, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, mm_encoder_cfg=mm_encoder_cfg, add_molecule_tokens=args.add_molecule_tokens)
    model = model.to(torch.bfloat16)
    # Sampling 
    batch_size = args.batch_size
    outs = []
    
    samples = 0
    for instructions, graphs, all_fg_atoms, gts in tqdm(
        iterate_test_files(args, convert_smiles_to_graph=True, batch_size=batch_size),
        total=_length_test_file(args)//batch_size,
    ):  
        bs = len(instructions)
        if args.task != "molecule_design":
            # graph_tensors = [_convert_dict_to_Data(graph).to(device) for graph in graphs]
            graph_tensors = [_convert_list_of_dict_to_Data(graph, device) for graph in graphs]
        else:
            graph_tensors = None
        if args.task != "molecule_design":
            all_fg_atoms = [all_fg_atom for all_fg_atom in all_fg_atoms]
        else:
            all_fg_atoms = None
            
        input_ids_batch = []
        stopping_criteria_batch = []
        for i in range(bs):
            qs = instructions[i]
            if args.task != "molecule_design":
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN * len(graph_tensors[i])+ '\n' + qs
                    # qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            # print(qs)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
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
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
                stopping_criteria=stopping_criteria_batch
            )
        # debug_s = '[C][N][Branch1][C][C][C][=O].[O].[O][=C][Branch1][C][O-1][O].[Na+1]'
        # debug_input_ids = tokenizer(debug_s).input_ids
        # debug_output_ids = tokenizer.decode(debug_input_ids).strip()
        # print(debug_input_ids)
        # print(debug_output_ids)
        # print('1: ',output_ids)
        # print('2: ',len(output_ids[0]))
        # print('stop_str: ',stop_str)
        outputs = [] # list of str
        for i in range(bs):
            # print('3: ',input_ids.shape[1])
            # output = tokenizer.decode(output_ids[i]).strip()
            # print('4.1: ',output)

            output = tokenizer.decode(output_ids[i, input_ids.shape[1]:]).strip()
            # print('4: ',output)
            
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            if args.task == "reagent_pred" or args.task == "retrosynthesis" or args.task == "forward_pred" or args.task == "molecule_design":
                output = output.replace(' ','')
            outputs.append(output)
        
        for instruction, gt, output in zip(instructions, gts, outputs):
            outs.append(
                {
                    "prompt": instruction,
                    "gt_self": gt,
                    "pred_self": output,
                }
            )
            if args.debug:
                print({"gt": gt, "outputs": output}, "\n")
        samples += bs
        # if samples > 20:
        # break
    
    # store result
    json.dump(outs, ans_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="retrosynthesis")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default="eval_result/answer.jsonl")
    parser.add_argument("--graph-checkpoint-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--add-selfies", action="store_true")
    parser.add_argument("--add_task_identifier", action="store_true")
    parser.add_argument("--add_molecule_tokens", action="store_true")
    args = parser.parse_args()
    main(args)


"""
TASK=retrosynthesis
EPOCH=5
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path checkpoints/Graph-LLaVA/graph-text-molgen/$TASK-llava-moleculestm-vicuna-v1-3-7b-finetune_lora \
    --in-file  /shared_space/caohe/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/retrosynthesis_test.json \
    --answers-file eval_result/moleculestm-$TASK-${EPOCH}ep.jsonl \
    --graph-checkpoint-path checkpoints/MoleculeSTM/molecule_model.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --add-selfies \
    --debug 
"""

"""
TASK=reagent_pred
EPOCH=5
python -m llava.eval.molecule_metrics.generate_sample 
    --task $TASK \
    --model-path checkpoints/Graph-LLaVA/graph-text-molgen/$TASK-llava-moleculestm-vicuna-v1-3-7b-finetune_lora \
    --in-file /shared_space/caohe/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/reagent_prediction_test.json \
    --answers-file eval_result/moleculestm-$TASK-${EPOCH}ep.jsonl \
    --graph-checkpoint-path checkpoints/MoleculeSTM/molecule_model.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --temperature 0.2 --top_p 1.0 \
    --add-selfies \
    --debug 
"""

"""
TASK=forward_pred
EPOCH=5
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path checkpoints/Graph-LLaVA/graph-text-molgen/$TASK-llava-moleculestm-vicuna-v1-3-7b-finetune_lora \
    --in-file /shared_space/caohe/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/forward_reaction_prediction_test.json \
    --answers-file eval_result/moleculestm-$TASK-${EPOCH}ep.jsonl \
    --graph-checkpoint-path checkpoints/MoleculeSTM/molecule_model.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --temperature 0.2 --top_p 1.0 \
    --add-selfies \
    --debug 
"""

"""
TASK=property_pred
EPOCH=5
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path checkpoints/Graph-LLaVA/graph-text-molgen/$TASK-llava-moleculestm-vicuna-v1-3-7b-finetune_lora \
    --in-file /cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/property_prediction_test.json \
    --answers-file eval_result/moleculestm-$TASK-${EPOCH}ep.jsonl \
    --graph-checkpoint-path checkpoints/MoleculeSTM/molecule_model.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --temperature 0.2 --top_p 1.0 \
    --add-selfies --debug 
"""