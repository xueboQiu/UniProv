import dgl
import numpy as np
import torch
from tqdm import tqdm

from train import add_loss
from utils.loaddata import transform_graph


def batch_level_train(model, graphs, train_loader, optimizer, main_args, n_dim=0, e_dim=0):
    epoch_iter = tqdm(range(main_args.max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for _, batch in enumerate(train_loader):
            batch_g = [transform_graph(graphs[idx][0], n_dim, e_dim).to(main_args.device) for idx in batch]
            batch_g = dgl.batch(batch_g)
            model.train()
            total_loss = model(batch_g)

            # add loss value to list
            add_loss(total_loss)
            # dynamic task priority
            losses = []
            for k, v in total_loss.items():
                losses.append(getattr(main_args, k.replace('loss', 'cof')) * v)
                # losses.append(v/sum(list(total_loss.values())) * v)
            losses = torch.stack(losses)
            # losses = torch.stack(list(total_loss.values()))
            # weights = dtp.update_weights(losses)
            # final_loss = (weights * losses).sum() /n_train
            final_loss = losses.sum()
            loss_list.append(final_loss.item())

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            del batch_g
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    return model
