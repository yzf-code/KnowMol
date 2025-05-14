#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import shutil

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from .multimodal_encoder.builder import build_graph_tower

def embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel
    ):
    embedding_layer = model.get_input_embeddings()
    print(f"ori Embedding layer shape: {embedding_layer.weight.shape}")
    num_new_tokens = len([line.strip() for line in open('/home/yangzaifei/MolLLM/codes/InstructMol/selfies_dict.txt')])
    # # num_new_tokens = tokenizer.add_tokens(selfies_dict_list)
    # new_token_ids = tokenizer.convert_tokens_to_ids(selfies_dict_list)
    model.resize_token_embeddings(len(tokenizer))
    # if num_new_tokens > 0:
    # input_embeddings = model.get_input_embeddings().weight.data
    # output_embeddings = model.get_output_embeddings().weight.data

    # input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
    #     dim=0, keepdim=True)
    # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
    #     dim=0, keepdim=True)

    # input_embeddings[-num_new_tokens:] = input_embeddings_avg
    # output_embeddings[-num_new_tokens:] = output_embeddings_avg

    embedding_layer = model.get_input_embeddings()
    print(f"new Embedding layer shape: {embedding_layer.weight.shape}")
    print(embedding_layer.weight[-num_new_tokens:].requires_grad)
    assert embedding_layer.weight[-num_new_tokens:].requires_grad, "New token embeddings are not trainable!"

    


def update_pretrained_config(pretrained_config, update_config):
    if isinstance(pretrained_config, dict):
        pretrained_config.update(update_config)
    else:
        config_class = pretrained_config.__class__
        cfg_pretrained_dict = pretrained_config.to_dict()
        cfg_pretrained_dict.update(**update_config)
        pretrained_config = config_class(**cfg_pretrained_dict)
    return pretrained_config


def load_pretrained_model(
    model_path, 
    model_base, 
    model_name, 
    load_8bit=False, 
    load_4bit=False, 
    device_map="auto",
    mm_encoder_cfg=None,
    add_molecule_tokens=False,
    **kwargs
):
    kwargs.update({"device_map": device_map})

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if mm_encoder_cfg is not None:
                lora_cfg_pretrained = update_pretrained_config(lora_cfg_pretrained, mm_encoder_cfg)
            if add_molecule_tokens:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            print(lora_cfg_pretrained)
            # currently changed to LlavaGraphLlamaForCausalLM
            model = LlavaGraphLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            # model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            # for name, param in model.named_parameters():
            #     print(name)
            #     if 'lm_head' in name:
            #         print(len(param))
            #         print(param)
            # graph_tower = model.get_graph_tower()
            # for name, param in graph_tower.named_parameters():
            #     print(name)
            #     print(param)
            #     break
            

            if mm_encoder_cfg is not None:
                model.model.graph_tower = build_graph_tower(lora_cfg_pretrained)
                graph_tower = model.get_graph_tower()
                graph_tower.to(dtype=torch.float16, device=model.device)
                
            # for name, param in graph_tower.named_parameters():
            #     print(name)
            #     print(param)
            #     break
                
            # new_graph_tower = build_graph_tower(lora_cfg_pretrained)
            # for name, param in new_graph_tower.named_parameters():
            #     print(name)
            #     print(param)
            #     break
            if add_molecule_tokens:
                print('add_molecule_tokens_and_embedding_resize')
                new_token_ids = embedding_resize(
                    tokenizer=tokenizer, model=model,)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                print('model.lm_head.weight.shape[0] != token_num')
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            # print(non_lora_trainables)
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
            # for name, param in model.named_parameters():
            #     print(name)
            #     if 'lm_head' in name:
            #         print(len(param))
            #         print(param)
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if mm_encoder_cfg is not None:
                    cfg_pretrained = update_pretrained_config(cfg_pretrained, mm_encoder_cfg)
                # currently changed to LlavaGraphLlamaForCausalLM
                model = LlavaGraphLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

                if mm_encoder_cfg is not None:
                    model.model.graph_tower = build_graph_tower(cfg_pretrained)
                    graph_tower = model.get_graph_tower()
                    graph_tower.to(dtype=torch.float16, device=model.device)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        # else:
        #     if 'mpt' in model_name.lower():
        #         tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        #         model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        #     else:
        #         tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        #         model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        
        if hasattr(model, 'get_vision_tower'):
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device='cuda', dtype=torch.float16)
            image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
