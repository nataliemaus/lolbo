import torch
import math
import selfies as sf
from torch.utils.data import TensorDataset, DataLoader


def update_models_end_to_end(
    train_x,
    train_y_scores,
    objective,
    model,
    mll,
    learning_rte,
    num_update_epochs
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    SELFIES VAE interface
    '''
    objective.vae.train()
    model.train() 
    optimizer = torch.optim.Adam([
            {'params': objective.vae.parameters()},
            {'params': model.parameters(), 'lr': learning_rte} ], lr=learning_rte)
    # max batch size smaller to avoid memory limit with longer strings (more tokens)
    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560/max_string_length)) 
    num_batches = math.ceil(len(train_x) / bsz)
    for _ in range(num_update_epochs):
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
            batch_list = train_x[start_idx:stop_idx]
            z, vae_loss = objective.vae_forward(batch_list)
            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float() 
            pred = model(z)
            surr_loss = -mll(pred, batch_y.cuda())
            # add losses and back prop 
            loss = vae_loss + surr_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.vae.parameters(), max_norm=1.0)
            optimizer.step()
    objective.vae.eval()
    model.eval()

    return objective, model


def update_surr_model(
    model,
    mll,
    learning_rte,
    train_z,
    train_y,
    n_epochs
):
    model = model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rte} ], lr=learning_rte)
    train_bsz = min(len(train_y),128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()

    return model

