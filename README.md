# Gaussian Mixture - variational graph auto-encoder(GMVAE) in Pytorch Geometric

This repository implements Gaussian Mixture - variational graph auto-encoder(GMVAE) in [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), adapted from the autoencoder example [code](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py)  in pyG. F

##

## Requirements

- Python >= 3.6
- Pytorch == 1.5
- Pytorch Geometric == 1.5
- scikit-learn
- scipy

## How to run
## First, replace the autoencoder.py with that in /Users/zhangzheng/anaconda3/lib/python3.11/site-packages/torch_geometric/nn/models/autoencoder.py in your pytorch geometric install files folder.

1. Configure the arguments in `config/vgae.yaml` file. You can also make your own config file.

2. Specify the config file and run the training script.
```
python train.py --load_config config/vgae.yaml
```

## Result
