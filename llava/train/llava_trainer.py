import os
import torch
import random
from transformers import Trainer
from typing import Optional
from torch.utils.data import WeightedRandomSampler, RandomSampler, Sampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer import has_length,is_datasets_available
from tqdm import tqdm
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

class SingleDataset_Sampler(Sampler):
    def __init__(self, concatdataset, batch_size):
        self.concatdataset = concatdataset
        self.dataset_sizes = [len(dataset) for dataset in concatdataset.datasets]
        self.batch_size = batch_size
        # 对每个数据集生成索引并分开保存
        self.dataset_indices = [list(range(sum(self.dataset_sizes[:i]), sum(self.dataset_sizes[:i+1]))) for i in range(len(self.concatdataset.datasets))]
        for dataset_indice in self.dataset_indices:
            random.shuffle(dataset_indice)
        
        # print(self.dataset_indices)
        
    def __iter__(self): 
        all_indices_split = []
        for indices in self.dataset_indices:
            # 只采样完整的 batch，丢弃不足 batch_size 的部分
            for i in range(0, len(indices), self.batch_size):
                if i + self.batch_size <= len(indices): 
                    all_indices_split.append(indices[i:i + self.batch_size])
        random.shuffle(all_indices_split)
        # print(all_indices)
        # self.all_indices = sum(all_indices,[])
        all_indices=[]
        for split in all_indices_split:
            all_indices += split
        # print(all_indices)
        return iter(all_indices)
    
    def __len__(self):
        # return sum(self.dataset_sizes)
        return sum(size // self.batch_size for size in self.dataset_sizes) * self.batch_size

class LLaVATrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        if hasattr(self.train_dataset, 'datasets') and len(self.train_dataset.datasets)>1:
            return SingleDataset_Sampler(self.train_dataset, self._train_batch_size)
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
