

### DDP Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_addr 127.0.0.34 --master_port 29503 train.py --cfg=configs/sourceOnly_syn2sk.yaml
```

### Train with single GPU
1. set --use_ddp to 0, i.e.,
`parser.add_argument('--use_ddp', type=int, default=0, help='using ddp to train model')`

### Infer
```
python gen_soft_SynLiDAR2SK_noDA.py --checkpoint_path logs/SynLiDAR2SK_M34_XYZ/2022-10-25-13_42/checkpoint_val_Sp.tar --result_dir SynLiDAR2SK_M34_XYZ_sp
```

### Eval performance
```
python eval_Folders_SemanticKITTI_npy_Tra.py --predictions SynLiDAR2SK_M34_XYZ_sp
```