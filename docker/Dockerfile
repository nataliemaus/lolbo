FROM pytorch/pytorch:latest 
RUN pip install fire 
RUN pip install rdkit-pypi
RUN pip install gpytorch 
RUN pip install botorch
RUN pip install guacamol
RUN pip install wandb
RUN pip install selfies
RUN pip install pytorch_lightning
RUN pip install pandas
RUN pip install networkx
RUN pip install matplotlib
RUN apt update
RUN apt install -y build-essential
RUN apt install -y libxrender1 libxext6 software-properties-common apt-utils
RUN conda install -y pomegranate
RUN pip install --no-deps molsets
RUN pip install fcd-torch
# ENV WANDB_API_KEY=$YOUR_WANDB_API_KEY_HERE
