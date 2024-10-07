# HyperPhS

## Title
HyperPhS: A pharmacophore-guided multi-view representation for metabolic stability prediction through contrastively hypergraph learning

## Abstract
Metabolic stability is crucial in the early stages of drug discovery and development. Pharmacophores, the functional groups within drug molecules, directly bind to receptors or biological macromolecules to produce biological effects, thereby significantly influencing metabolic stability. Accurately predicting this stability can streamline drug candidate screening and optimize lead compounds. Given the high costs of wet lab experiments, in silico prediction offers a valuable yet underdeveloped alternative. Furthermore, determining metabolic stability using a pharmacophore-guided approach remains a significant challenge.

To address these issues, we develop a novel pharmacophore-guided hypergraph-based approach for metabolic stability prediction named HyperPhS.
![image](model.png)

## Setup
Please install HyperPhS in a virtual environment to ensure it has conflicting dependencies.
```
Python == 3.8
PyTorch == 2.0
scikit-learn == 1.2.2
pandas == 2.0.2
numpy == 1.23.5
RDKit == 2023.03.1
network == 2.8.4
PyG == 2.3.1
Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
```

## Training the model

To train the model, you will use the `train.py` script. This script accepts several command-line arguments to customize the training process.

## Training command:

Run the following command to start the training process:

``` bash
$ python train_singleGPU.py --known_class 'True' --checkpoint_dir 'checkpoint' --device 'cuda:0'
```
Replace the argument values as per your requirements. For instance, use `--device`  cpu if you're training on a CPU.

## Multi-GPU Training command：

To run the training on multiple GPUs, use the following command:

``` bash
$ CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 train_multiGPU.py --known_class 'True' --checkpoint_dir 'checkpoint'
```

* `CUDA_VISIBLE_DEVICES=0,2`: This specifies the GPU devices (in this case, GPUs 0 and 2) on which the training will be run.
* `--nproc_per_node=2`: This indicates the total number of GPUs to be used (2 GPUs in this example).
* `train_multiGPU.py`: This is the training script.

You can also add additional parameters to train_multiGPU.py to adjust whether you are running known reaction types or unknown reaction types. The parameters are the same as those used in the single GPU setup.

## Validating the model

After training, you can validate the model's accuracy using the `translate.py` script on testing set.
* `--known_class`: As in the training step, this indicates whether the class is known or unknown.
* `--checkpoint_dir`: The directory where your trained model checkpoints are stored.
* `--checkpoint`: The specific checkpoint file to use for validation. Replace `{training_step}` with the appropriate training step number. We provide an example checkpoint trained on uspto50k datasets. You can download the checkpoint [here](https://drive.google.com/drive/folders/12gNpyfM6zZJlaoHsL-2-Jwmt3qoU1_om?usp=sharing).
* `--device`: The device to run the validation on, either GPU (cuda:0) or CPU (cpu).

``` bash
$ python translate.py --known_class 'False' --checkpoint_dir 'checkpoint' --checkpoint 'a_model_{training_step}.pt' --device 'cuda:0'
```

## Perform the retrosynthesis step
After the training is completed, you can run the inference.py for one-step retrosynthesis prediction
* `--beam_size`: The top k predictions for a molecule 
``` bash
$ python inference.py --smiles 'Clc1cc(Cl)c(CBr)cn1' --beam_size 10 --checkpoint_dir 'checkpoint' --checkpoint 'unknown_model.pt'
```
