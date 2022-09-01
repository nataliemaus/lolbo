# Local Latent Bayesian Optimization (LOLBO)
Official implementation of Local Latent Space Bayesian Optimization over Structured Inputs https://arxiv.org/abs/2201.11872

This repository includes base code to run LOLBO, along with full implementation for GuacaMol and Penalized logP Molecular Optimization tasks to replicate results in Figure 1 left and Figure 2 in the original paper. The base code is also set up to allow LOLBO to be run on any other latent space optimization task. Please contact the authors if you need any help with applying the LOLBO to run the other two tasks from the paper (Arithmetic Expressions, DRD3 Receptor Docking Affinity), or any other latent space optimization task. 

## Weights and Biases (wandb) tracking
This repo it set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site
Otherwise, the code can also be run without wandb tracking by simply setting the argument `--track_with_wandb False` (see example commands below). 

## Getting Started

### Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must therefore be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

### Environment Setup (Conda)
Follow the steps below to install all dependencies to run LOLBO. Execute in the repository ROOT.

```Bash
conda env create -f conda_env.yml
conda activate lolboenv
pip install molsets --no-deps
```

The resultant environment will have all imports necessary to run LOLBO on the example GuacaMol and Penalized Log P molecular optimization tasks in this repo.

## How to Run LOLBO on Our Example Molecular Optimization Tasks

In this section we provide commands that can be used to start a LOLBO optimization after the environment has been set up. 

### Args:

To start an molecule optimization run, run `scripts/molecule_optimization.py` with desired command line arguments. To get a list of command line args specifically for the molecule optimization tasks with the SELFIES VAE, run the following: 

```Bash
cd scripts/

python3 molecule_optimization.py -- --help
```

For a list of the remaining possible args that are the more general LOLBO args (not specific to molecule tasks) run the following:

```Bash
cd scripts/

python3 optimize.py -- --help
```

The above commands will give defaults for each arg and a description of each. The only required argument is `--task_id`, which is the string id that determines the optimization task. 

### Task IDs
#### GuacaMol Task IDs
This code base provides support for the following 12 GuacaMol Optimization Tasks:

| task_id | Full Task Name     |
|---------|--------------------|
|  med1   | Median molecules 1 |
|  med2   | Median molecules 2 |
|  pdop   | Perindopril MPO    |
|  osmb   | Osimertinib MPO    |
|  adip   | Amlodipine MPO     |
|  siga   | Sitagliptin MPO    |
|  zale   | Zaleplon MPO       |
|  valt   | Valsartan SMARTS   |
|  dhop   | Deco Hop           |
|  shop   | Scaffold Hop       |
|  rano   | Ranolazine MPO     |
|  fexo   | Fexofenadine MPO   |

The original LOLBO paper features results on zale, pdop, and rano. For descriptions of these and the other GuacaMol objectives listed, as well as a leaderboard for each of these tasks, see https://www.benevolent.com/guacamol

#### Penalized Log P Task ID
To run on Penalized Log P instead of one of the above GuacaMOl tasks, use the following four-letter task id:

```
logp --> Penalized Log P
```

### Example Command to Optimize Penalized Log P with LOLBO 
##### (Replicates Result in Figure 1 Left of Paper)

```Bash
cd scripts/

CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id logp --max_n_oracle_calls 500 --bsz 1 --k 10 - run_lolbo - done 
```
#### Command Modified to Run With Weights and Biases (wandb) Tracking
```Bash
CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id logp --max_n_oracle_calls 500 --bsz 1 --k 10 --track_with_wandb True --wandb_entity $YOUR_WANDB_USERNAME - run_lolbo - done 
```

### Example Command to GuacaMol Objectives with LOLBO
To replicate the result in Figure 2 of the paper, simply run the below command three times - once each with `--task_id pdop`, `--task_id rano`, and `--task_id zale` (modify as above to run with wandb tracking):

```Bash
cd scripts/

CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id zale --max_string_length 400 --max_n_oracle_calls 120000 --bsz 10 - run_lolbo - done 
```

## How to Run LOLBO on Other Tasks
To run LOLBO on other tasks, you'll need to write two new classes: 

1. Objective Class

Create a new child class of LatentSpaceObjective (see `lolbo/latent_space_objective.py`) which all outlined methods defined. 

See example objective class for molecule tasks: 
`lolbo/molecule_objective.py `

2. Top Level Optimization Class

Create a new child class of Optimize (see `scripts/optimize.py`) which defines the two methods specific to the optimization task. The first is the `initialize_objective` method, which defines the variable `self.objective` to be an object of the new objective class created in step 1. The second is the `load_train_data` method which loads in the available training data that will be used to initialize the optimization run. 

See example optimization class for molecule tasks: 
`scripts/molecule_optimization.py`
