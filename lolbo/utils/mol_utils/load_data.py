import numpy as np
import pandas as pd
import torch
import math


def load_molecule_train_data(
    task_id,
    path_to_vae_statedict,
    num_initialization_points=10_000,
): 
    df = pd.read_csv("../lolbo/utils/mol_utils/guacamol_data/guacamol_train_data_first_20k.csv")
    df = df[0:num_initialization_points]
    train_x_smiles = df['smile'].values.tolist()
    train_x_selfies = df['selfie'].values.tolist() 
    train_y = torch.from_numpy(df[task_id].values).float() 
    train_y = train_y.unsqueeze(-1)
    train_z = load_train_z(
        num_initialization_points=num_initialization_points,
        path_to_vae_statedict=path_to_vae_statedict
    ) 

    return train_x_smiles, train_x_selfies, train_z, train_y


def load_train_z(
    num_initialization_points,
    path_to_vae_statedict,
):
    state_dict_file_type = path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
    path_to_init_train_zs = path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
    # if we have a path to pre-computed train zs for vae, load them
    try:
        zs = pd.read_csv(path_to_init_train_zs, header=None).values
        # make sure we have a sufficient number of saved train zs
        assert len(zs) >= num_initialization_points
        zs = zs[0:num_initialization_points]
        zs = torch.from_numpy(zs).float()
    # otherwisee, set zs to None 
    except: 
        zs = None 

    return zs


def compute_train_zs(
    mol_objective,
    init_train_x,
    bsz=64
):
    init_zs = []
    # make sure vae is in eval mode 
    mol_objective.vae.eval() 
    n_batches = math.ceil(len(init_train_x)/bsz)
    for i in range(n_batches):
        xs_batch = init_train_x[i*bsz:(i+1)*bsz] 
        zs, _ = mol_objective.vae_forward(xs_batch)
        init_zs.append(zs.detach().cpu())
    init_zs = torch.cat(init_zs, dim=0)
    # now save the zs so we don't have to recompute them in the future:
    state_dict_file_type = mol_objective.path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
    path_to_init_train_zs = mol_objective.path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
    zs_arr = init_zs.cpu().detach().numpy()
    pd.DataFrame(zs_arr).to_csv(path_to_init_train_zs, header=None, index=None) 

    return init_zs
