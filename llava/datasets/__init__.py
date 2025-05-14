from .lazy_supervised_dataset import LazySupervisedDataset, LazySupervisedGraphDataset
from .reagent_pred_dataset import ReagentPredSupervisedGraphDataset
from .forward_pred_dataset import ForwardPredSupervisedGraphDataset
from .retrosynthesis_dataset import RetrosynthesisSupervisedGraphDataset
from .property_pred_dataset import PropertyPredSupervisedGraphDataset
from .mol_design_dataset import MolDesignSupervisedGraphDataset
from .collators import DataCollatorForSupervisedDataset, GraphDataCollatorForSupervisedDataset
from .MoleculeNet_classification_dataset import MoleculeNetSupervisedGraphDataset
from torch.utils.data import ConcatDataset
from .knowmol_dataset import KnowMolDataset


def build_dataset(tokenizer, data_args):
    data_type = data_args.data_type
    if data_type == "supervised":
        dataset = LazySupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "reagent_pred":
        dataset = ReagentPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "forward_pred":
        dataset = ForwardPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "retrosynthesis":
        dataset = RetrosynthesisSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "property_pred":
        dataset = PropertyPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "molecule_design":
        dataset = MolDesignSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "pretrain_2_task":
        smiles_2_l1_2_l2_2_l3_2_l4_data = KnowMolDataset(
            data_path="/home/yangzaifei/MolLLM/data/PubChem/PubChem_text/KnowMol_datas/Multi_Round_Question_Answering.pkl",
            tokenizer=tokenizer,
            data_args=data_args,
            sample_weight = 5
        )
        l1_l2_l3_l4_2_smiles_data = KnowMolDataset(
            data_path="/home/yangzaifei/MolLLM/data/PubChem/PubChem_text/KnowMol_datas/Description_Guided_Molecule_Generation.pkl",
            tokenizer=tokenizer,
            data_args=data_args,
            sample_weight = 5
        )
        dataset = ConcatDataset([
            smiles_2_l1_2_l2_2_l3_2_l4_data, 
            l1_l2_l3_l4_2_smiles_data
            ])
    else:
        raise NotImplementedError(f"Unknown data type: {data_type}")
    return dataset