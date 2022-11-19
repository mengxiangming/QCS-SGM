# ICLR2023 submission  #3904: Quantized Compressed Sensing with Score-Based Generative Models

This repo contains the Pytorch implementation for the ICLR2023 submission #3904 "Quantized Compressed Sensing with Score-Based Generative Models"
-----------------------------------------------------------------------------------------


![samples](assets/ffhq_123bit.png)

(Our results on FFHQ 256px high-resolution images with 8x noisy heavily quantized (1-bit, 2-bit, and 3-bit)  measurements y = Q(Ax + n), where A is a Gaussian measurement matrix. The original dimension of signal x is N = 196608, while the number of measurements is 8x, i.e., M = 24576 << N. An additive Gaussian noise n with standard deviation  0.001 is added.)

## Running Experiments

### Dependencies

Create a new environment and run the following to install all necessary python packages for our code.

```bash
pip install -r requirements.txt
```

### Project structure

main.py` is the file that you should run for quantized CS. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --model_dir DIR
               [--comment COMMENT] [--verbose VERBOSE] [--test] [--sample]
               [--fast_fid] [--resume_training] [-i IMAGE_FOLDER] [--ni]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --model_dir DIR       Path for putting the checkpoint file.
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  --test                Whether to test the model
  --sample              Whether to produce samples from the model
  --fast_fid            Whether to do fast fid test
  --resume_training     Whether to resume training
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples (saved results)
  --ni                  No interaction. Suitable for Slurm Job launcher
```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
├── logs # contains checkpoints and samples produced during training
│   └── <model_dir> # a folder named by the argument `--model_dir` specified to main.py
│      ├── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
├── cs_results # contains original/recovered images in CS
│   └── DATA_Type # Name of datasets, e.g., MNIST, CIFAR10, CELEBA, FFHQ
│       └── 1-bit # recovered images from 1-bit CS       
```       └── image_x.png # samples generated from checkpoint_x.pth 
│       └── 2-bit # recovered images from 2-bit CS         
```       └── image_x.png # samples generated from checkpoint_x.pth
│       └── 3-bit # recovered images from 3-bit CS        
```       └── image_x.png # samples generated from checkpoint_x.pth
│       └── linear # recovered images from linear (un-quantized) CS        
```       └── image_x.png # samples generated from checkpoint_x.pth
```




### Getting Started 
To reconstruct images from the Q-bit quantized noisy measurements using NCSNv2, one can run the code as follows:
(Take CelebA dataset for an example)

Step 1: 
Edit `celeba.yml` in ./configs/ to specify the simulated setting, e.g.,

In the sampling group of `**.yml`
    checkpoint id:  `ckpt_id` 
    learning rate:  `step_lr`
    whether the CS problem is considered:  `linear_inverse`  (True or False, set true for CS)
    True: perform conditional sampling based on observations y
    False: perform unconditional sampling 

In the measurements group:
Number of measurements M : `measure_size` 
additive noise variance sigma^2 : `noise_variance`  
Whether or not quantization is used: `quantization`  (True or False)
Number of quantization bits Q: `quantize_bits`


Step 2: 
Run the following command 
```shell
python python main.py  --sample --config celeba.yml --model_dir celeba --i ./cs_results
```
Reconstructed  will be saved in `<exp>/celeba_demo_results/`.



## Pretrained Checkpoints

Please download the open-sourced pretrained checkpoints from the following link for Cifar10, CelebA, and FFHQ, and put them in the 
./exp/logs/cifar10, ./exp/logs/celeba, and ./exp/logs/ffhq, respectively (Or, you can simply download the whole exp.zip file, unzip it in the root folder of this project). Please select the pre-trained models with the specified `ckpt_id`  in the  config files

Link: https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing


It assumes the `--exp`   argument is set to `exp`


Notice:

There is no pre-trained checkpoint for MNIST in the above link. Ii our work, we trained on MNIST ourselves with the `configs/mnist.yml`  in this project using the open-sourced ncsnv2 code:
Link: https://github.com/ermongroup/ncsnv2
Please follow exactly their descriptions to train on MNIST. We did not share the checkpoint since it is larger than 100MB. 

If you do not want to train by yourself, you can simply have a try on celeba, cifar10, and ffhq with pre-trained checkpoints. 

## References

This repo is built on top of the open-sourced ncsnv2 code: https://github.com/ermongroup/ncsnv2



