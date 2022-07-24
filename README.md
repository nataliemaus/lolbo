# Local Latent Bayesian Optimization (LOLBO)
Official implementation of Local Latent Space Bayesian Optimization over Structured Inputs https://arxiv.org/abs/2201.11872

This repository includes base code to run LOLBO, along with full implemention for GuacaMol and Penalized logP Molecular Optimization tasks to replicate results in Figure 1 left and Figure 2 in the original paper. The base code is also set up to allow LOLBO to be run on any other latent space optimization task. Please contact the autors if you need any help with applying the LOLBO to run the other two tasks from the paper (Arithmetic Expressions, DRD3 Receptor Docking Affinity), or any other latent space optimization task. 

## Weights and Biases (wandb) tracking
This repo it set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site
Otherwise, the code can also be run without wandb tracking by simply setting the argument --track_with_wandb False (see example commands below). 

## Environment Setup 

### Option A: Docker 
If you have a docker account, we recommend using the provided Dockerfile (see docker/Dockerfile) to build an docker image with all required packages to run the code. 

cd docker/

docker build -t YOUR_DOCKER_USERNAME/lolbo .

If you don't have an account you can also create one here:
https://www.docker.com/

If you have a Weights and Biases (wandb) account, we also recommend uncommenting the last line in the docker file and replacing $YOUR_WANDB_API_KEY_HERE with your own wandb account API key. This way you will not have to re-login to your wandb account each time you boot up the docker image. If you have a wandb account, your API key can be found here:
https://wandb.ai/authorize 

Note that the Dockerfile has all imports necessary to run LOLBO on the example GuacaMol and Penalized Log P molecular optimization tasks in this repo. If you would like to apply LOLBO to other optimization tasks with additional package requirements, additional install statements can be added to the Dockerfile as needed.

### Option B: Conda
Follow the steps below to install all dependencies to run LOLBO:

conda create --name lolbo

conda activate lolbo 

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install gpytorch 

pip install botorch

pip install wandb

pip install fire 

pip install pandas

pip install pytorch-lightning

pip install matplotlib

###### Additional packages specificly for molecule optimization tasks:

pip install guacamol

pip install selfies

pip install rdkit-pypi

pip install networkx

pip install --no-deps molsets

pip install fcd-torch

pip install pomegranate

## How to Run LOLBO on Our Example Molecular Optimization Tasks

In this section we provide commands that can be used to start a LOLBO optimization after the environment has been set up. 

### Args:

To start an molecule optimization run, run scripts/molecule_optimization.py with desired command line arguments. To get a list of command line args specifically for the molecule optimization tasks with the SELFIES VAE, run the following: 

cd scripts/

python3 molecule_optimization.py -- --help

For a list of the remaining possible args that are the more general LOLBO args (not specific to molecule tasks) run the following:

cd scripts/

python3 optimize.py -- --help

The above commands will give defaults for each arg and a description of each. The only required argumet is --task_id, which is the string id that determines the optimization task. 

### Task IDs
#### GuacaMol Task IDs
This code base provides support for the following 12 GuacaMol Optimization Tasks:
###### task_id   --> Full Task Name
med1 --> Median molecules 1

med2 --> Median molecules 2

pdop --> Perindopril MPO

osmb --> Osimertinib MPO

adip --> Amlodipine MPO 

siga --> Sitagliptin MPO

zale --> Zaleplon MPO

valt --> Valsartan SMARTS

dhop --> Deco Hop

shop --> Scaffold Hop

rano --> Ranolazine MPO 

fexo --> Fexofenadine MPO

The original LOLBO paper features resuls on zale, pdop, and rano. For descriptions of these and the other GuacaMol objectives listed, as well as a leaderboard for each of these tasks, see https://www.benevolent.com/guacamol

#### Penalized Log P Task ID
To run on Penalized Log P instead of one of the above GuacaMOl tasks, use the following four-letter task id:

logp --> Penalized Log P

### Example Command to Optimize Penalized Log P with LOLBO 
##### (Replicates Result in Figure 1 Left of Paper)

cd scripts/

CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id logp --max_n_oracle_calls 500 --bsz 1 --k 10 - run_lolbo - done 

#### Command Modified to Run With Weights and Biases (wandb) Tracking
CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id logp --max_n_oracle_calls 500 --bsz 1 --k 10 --track_with_wandb True --wandb_entity $YOUR_WANDB_USERNAME - run_lolbo - done 

### Example Command to GuacaMol Objectives with LOLBO
##### To replicate the result in Figure 2 of the paper, simply run the below command three times - once each with --task_id pdop, --task_id rano, and --task_id zale (modify as above to run with wandb tracking):

cd scripts/

CUDA_VISIBLE_DEVICES=1 python3 molecule_optimization.py --task_id zale --max_string_length 400 --max_n_oracle_calls 120000 --bsz 10 - run_lolbo - done 

## How to Run LOLBO on Other Tasks
To run LOLBO on other tasks, you'll need to write two new classes: 

1. Objective Class

Create a new child class of LatentSpaceObjective (see lolbo/latentspaceobjective.py) which all outlined methods defined. 

See example objective class for molecule tasks: 
lolbo/moleculeobjective.py 

2. Top Level Optimization Class

Create a new child class of Optimize (see scripts/optimize.py) which defines the two methods specific to the optimization task. The first is the initialize_objective method, which defines the variable self.objective to be an object of the new objective class created in step 1. The second is the load_train_data method which loads in the available training data that will be used to initialize the optimization run. 

See example optimization class for molecule tasks: 
scripts/molecule_optimization.py
