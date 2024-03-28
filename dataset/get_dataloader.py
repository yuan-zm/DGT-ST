from torch.utils.data import DataLoader
import numpy as np
import torch
from dataset.dataProvider import DataProvider


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_TV_dl(cfg, train_dataset, val_dataset, domain='source'):

  
    if domain == 'source':
        train_source_loader = DataProvider(
            train_dataset,
            batch_size=cfg.DATALOADER.TRA_BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=my_worker_init_fn,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_source_loader = DataLoader(
            train_dataset,
            batch_size=cfg.DATALOADER.TRA_BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=my_worker_init_fn,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            drop_last=True
        )
    val_source_loader = DataLoader(
        val_dataset,
        batch_size=cfg.DATALOADER.VAL_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=my_worker_init_fn,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
   
    return train_source_loader, val_source_loader#, train_sampler, valid_sampler
