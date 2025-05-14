# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import transformers

from llava.datasets import GraphDataCollatorForSupervisedDataset, build_dataset
from llava.train.llava_trainer import LLaVATrainer
from llava import conversation as conversation_lib
from llava.model import *


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    only_tune_mm_mlp_adapter: bool = field(default=False)
    graph_tower: Optional[str] = field(default=None)
    gin_num_layers: int = field(default=5)
    gin_hidden_dim: int = field(default=300)
    drop_ratio: float = field(default=0.1)
    graph_pooling: str = field(default='mean')
    init_checkpoint: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    previous_finetune_weights: Optional[str] = field(default=None)
    add_molecule_tokens: bool = field(default=False)
    train_molecule_tokens: bool = field(default=True)

@dataclass
class DataArguments:
    data_type: str = field(default="supervised")
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data.(.pkl)"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    add_task_identifier: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_graph_tower: bool = field(default=True)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 256
    lora_dropout: float = 0.1
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "only_tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)
        

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def add_molecule_tokens_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel
    ):
    embedding_layer = model.get_input_embeddings()
    rank0_print(f"ori Embedding layer shape: {embedding_layer.weight.shape}")
    selfies_dict_list = [line.strip() for line in open('/home/yangzaifei/MolLLM/codes/InstructMol/selfies_dict.txt')]
    num_new_tokens = tokenizer.add_tokens(selfies_dict_list)
    new_token_ids = tokenizer.convert_tokens_to_ids(selfies_dict_list)

    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        embedding_layer = model.get_input_embeddings()
        rank0_print(f"new Embedding layer shape: {embedding_layer.weight.shape}")
        rank0_print('New token embeddings grads', embedding_layer.weight[-num_new_tokens:].requires_grad)
        rank0_print('All token embeddings grads', embedding_layer.weight.requires_grad)
        assert embedding_layer.weight[-num_new_tokens:].requires_grad, "New token embeddings are not trainable!"

    return new_token_ids


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = build_dataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = GraphDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    rank0_print(model_args)
    rank0_print(data_args)
    rank0_print(training_args)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.graph_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            # here
            model = LlavaGraphLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )


    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
            
    embedding_layer = model.get_input_embeddings()
    rank0_print(f"Embedding layer shape: {embedding_layer.weight.shape}")
    if model_args.add_molecule_tokens:
        new_token_ids = add_molecule_tokens_and_embedding_resize(
            tokenizer=tokenizer, model=model,)
        num_new_tokens = len(new_token_ids)
        rank0_print(embedding_layer.weight[-num_new_tokens:].requires_grad)


    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        if model_args.previous_finetune_weights == None:
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding initial LoRA adapters...")
            model = get_peft_model(model, lora_config)
        else:
            rank0_print('Loading non_lora_trainables weights...')
            if os.path.exists(os.path.join(model_args.previous_finetune_weights, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_args.previous_finetune_weights, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_args.previous_finetune_weights, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            #设置这里不load embed
            model.load_state_dict(non_lora_trainables, strict=False)
            from peft import PeftModel
            # from peft.tuners.lora import mark_only_lora_as_trainable
            rank0_print("Adding pretrained LoRA adapters...")
            # model = PeftModel.from_pretrained(model, model_args.previous_finetune_weights)
            lora_config_new = LoraConfig(
                r=training_args.lora_r, # assert this is same with pretrain lora_r
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config_new)
            model.load_adapter(model_args.previous_finetune_weights,adapter_name='default')

            #         rank0_print('training_args.lora_dropout: ',training_args.lora_dropout)
            # # 修改 LoRA 的 dropout 参数
            # for module in model.modules():
            #     if hasattr(module, "lora_dropout"):
            #         module.lora_dropout.p = training_args.lora_dropout
            # for module in model.modules():
            #     if hasattr(module, "lora_dropout"):
            #         rank0_print(f"new LoRA dropout value: {module.lora_dropout.p}")

            # mark_only_lora_as_trainable(model)
            for name, param in model.named_parameters():
                if 'lora' in name or 'Lora' in name:
                    param.requires_grad = True
            # see: https://github.com/huggingface/peft/issues/184
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True

    if model_args.graph_tower is not None:
        model.get_model().initialize_graph_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        graph_tower = model.get_graph_tower()
        graph_tower.to(dtype=torch.float16, device=training_args.device)
        data_args.is_multimodal = True

        model.config.only_tune_mm_mlp_adapter = training_args.only_tune_mm_mlp_adapter = model_args.only_tune_mm_mlp_adapter
        if model_args.only_tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        
        if training_args.freeze_graph_tower:
            for name, param in model.named_parameters():
                if 'graph_tower' in name:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if 'graph_tower' in name:
                    param.requires_grad = True

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_graph_tokenizer(model_args, tokenizer=tokenizer)


    # if model_args.add_molecule_tokens:
    #     new_token_ids = add_molecule_tokens_and_embedding_resize(tokenizer=tokenizer, model=model)
        #设置这里load embed
        
    if model_args.add_molecule_tokens and model_args.train_molecule_tokens:
        # num_new_tokens = len(new_token_ids)
        # for idx in new_token_ids:
        #     model.get_input_embeddings().weight[idx].requires_grad = True
        #     rank0_print(model.get_input_embeddings().weight[idx].requires_grad)
        # rank0_print(model.get_input_embeddings().weight.requires_grad)
        # rank0_print(model.get_input_embeddings().weight[-num_new_tokens:].requires_grad)
        # model.get_input_embeddings().weight[-num_new_tokens:].requires_grad = True  # 明确设置为可训练
        # rank0_print(model.get_input_embeddings().weight[-num_new_tokens:].requires_grad)
        # model.get_input_embeddings().weight.data[-num_new_tokens:].requires_grad = True  # 明确设置为可训练
        # rank0_print(model.get_input_embeddings().weight[-num_new_tokens:].requires_grad)
        rank0_print('modify input layer')
        rank0_print(model.get_input_embeddings().weight.requires_grad)
        rank0_print(model.get_input_embeddings().weight[-num_new_tokens:].requires_grad)
        # model.get_input_embeddings().weight.requires_grad = True  # 明确设置为可训练
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
        rank0_print(model.get_input_embeddings().weight.requires_grad)
        rank0_print(model.get_input_embeddings().weight[-num_new_tokens:].requires_grad)

        rank0_print('modify output layer')
        rank0_print(model.get_output_embeddings().weight.requires_grad)
        rank0_print(model.get_output_embeddings().weight[-num_new_tokens:].requires_grad)
        # model.get_input_embeddings().weight.requires_grad = True  # 明确设置为可训练
        for param in model.get_output_embeddings().parameters():
            param.requires_grad = True
        rank0_print(model.get_output_embeddings().weight.requires_grad)
        rank0_print(model.get_output_embeddings().weight[-num_new_tokens:].requires_grad)


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    # save callback 
    from transformers import TrainerCallback
    class SaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(state.global_step))
            if args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                if args.local_rank in [-1, 0]:
                    model.config.save_pretrained(checkpoint_dir)
                    model.save_pretrained(checkpoint_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(checkpoint_dir, 'non_lora_trainables.bin'))
    # summary model parameter size
    if training_args.local_rank in [-1, 0]:
        from transformers import logging
        logging.set_verbosity_info()
        # print(model.num_parameters())
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)
            
    model.print_trainable_parameters()
    
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=[SaveCallback()],
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
