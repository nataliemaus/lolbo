import numpy as np
import pandas as pd
import torch


def load_molecule_train_data(
    task_id,
    data_folder,
    num_initialization_points,
): 
    train_z = load_train_z(data_folder) 
    train_x = load_all_decoded_smiles_list(data_folder)[0:len(train_z)] 
    train_y = load_train_y(task_id, data_folder)

    if len(train_z) < len(train_y):
        train_y = train_y[0:len(train_z)]
    else:
        train_z = train_z[0:len(train_y)]
    
    train_x = train_x[0:len(train_y)] 
    train_z = train_z[np.logical_not(np.isnan(train_y))]
    train_x = np.array(train_x)
    train_x = train_x[np.logical_not(np.isnan(train_y))]
    train_x = train_x.tolist()
    train_y = train_y[np.logical_not(np.isnan(train_y))]
    train_z = torch.from_numpy(train_z).float()
    train_y = torch.from_numpy(train_y).float() 

    if num_initialization_points != -1:
        print(f"Initializing with {num_initialization_points} points from training set")
        train_z = train_z[0:num_initialization_points] 
        train_y = train_y[0:num_initialization_points]
        train_x = train_x[0:num_initialization_points] 

    train_y = train_y.unsqueeze(-1)
    return train_x, train_z, train_y



def load_all_decoded_smiles_list(folder): 
    smiles_path = folder + "decoded_smiles.csv"
    smiles = pd.read_csv(smiles_path, header=None).values.squeeze().tolist()
    return smiles


def load_train_z(folder):
    zs_path1 = folder + "train_zs_first_third.csv"
    zs1 = pd.read_csv(zs_path1, header=None).values
    zs_path2 = folder + "train_zs_second_third.csv"
    zs2 = pd.read_csv(zs_path2, header=None).values
    zs_path3 = folder + "train_zs_third_third.csv"
    zs3 = pd.read_csv(zs_path3, header=None).values
    zs = np.vstack((zs1, zs2, zs3))
    return zs


def load_train_y(task_id, folder):
    score_array = pd.read_csv(folder + "train_ys_" + task_id + ".csv", header=None).values.squeeze()
    return score_array
