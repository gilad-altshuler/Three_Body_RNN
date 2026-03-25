# Beyond Linear Summation - Three Body RNNs [![DOI](https://zenodo.org/badge/681632750.svg)](https://doi.org/10.5281/zenodo.17208622)

## About
Code repository accompaning the paper "Beyond linear summation: Three-Body RNNs for modeling complex neural and biological systems"

<img width="2250" height="1500" alt="three_body_interactions" src="https://github.com/user-attachments/assets/3fe1f2a9-06ed-4d45-8a2f-c0b552270112" />

Fig 1. **Motivation: Three-body interactions.** 
(a) Dendritic nonlinearities, where inputs on the same branch gate each other. 
(b) Neuromodulatory axon operates as a third body in the synapse. 
(c) Incorporating glial cells into the network induces nonlinear summation. 
(d) Gene expression dimerization networks. Monomers form homo- or hetero-dimers that act as transcription factors (TFs). In the competitive setup shown, homo-dimers ($ii,jj,kk,\dots$) upregulate production of their own monomer. Hetero-dimers exert no direct regulation but lower the pool of free monomers, thereby indirectly downregulating expression.
(e) Traditional neural network with linear pre-synaptic summation (left) while biophysics is more complex and input couplings present (right).
    Figure created with BioRender.com. Panel (d) schematic inspired by Zhu et al., 2022.

## Reproduce paper figures 

**You may open the notebooks in google colab and simply click "Run all".**

|Figure        | Reproduce + Link to instructions |
|----------------------|------|
|Fig 1. Motivation: Three-body interactions|Created manually in Biorender.com|
|Fig 2. Neuroscience tasks and their biological gene expression counterparts|Created manually in Biorender.com|
|Fig 3. Low rank inference for TBRNNs|[`1_Low_rank_TBRNN_validation.ipynb`](#Low-rank-TBRNN-validation)|
|Fig 4. Expanding solution space|[`2_Solution_space.ipynb`](#Expanding-solution-space)|
|Fig 5. Synthetic validation of interaction-order detection|[`3_Teacher_Student.ipynb`](#Teacher-student-setup-on-synthetic-neuroscience-data)|
|Fig 6. Detection of higher-order interactions in the MultiFate dynamical system|[`4_MultiFate inference.ipynb`](#Multi-Fate-inference-task)|
|Fig 7. Interaction-order detection on neural population datasets|[`5_Neural_recordings_inference.ipynb`](#Monkey-neural-trajectory-inference-tasks)|
|Fig 8. Task-space mapping based on latent rank and interaction order|Created manually|
|Fig SI.1. Comparison of LrTBRNN with truncated $\Delta\mathcal{W}$|[`1_Low_rank_TBRNN_validation.ipynb`](#Low-rank-TBRNN-validation)|
|Fig SI.2. Distribution of pairwise CKA values between model classes|[`2_Solution_space.ipynb`](#Expanding-solution-space)|

## Results - how to run
### Start
To start please create a conda environment by:
```
cd Three_Body_RNN
conda env create -f TBRNN_env.yaml
conda activate TBRNN_env
```
(on Linux)

### Low-rank TBRNN validation
To reproduce the low-rank and theory validation data - run train script (you can also add nohup):
```
training_scripts/validation/run_multiple.sh > ../master_log.txt 2>&1 &
```
Next, run the notebook:

[1_Low_rank_TBRNN_validation.ipynb](notebooks/1_Low_rank_TBRNN_validation.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that notebook can be run independently without the reproduction train - data used already located in data/validation directory

### Expanding solution space
To reproduce - run train script (you can also add nohup):
```
bash training_scripts/solution_space/run_multiple.sh > ../master_log.txt 2>&1 &
```
Next, run the notebook:

[2_Solution_space.ipynb](notebooks/2_Solution_space.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that notebook can be run independently without the reproduction train - data used already located in data/solution_space directory

### Teacher-student setup on synthetic neuroscience data
To reproduce - run train script (you can also add nohup):
```
bash training_scripts/teacher_student/run_multiple_tasks.sh > ../master_log.txt 2>&1 &
```
Next, run the notebook:

[3_Teacher_Student.ipynb](notebooks/3_Teacher_Student.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that train in this task requires many gpus / time expensive, since we set the default to 30 runs per task and 6 ranks models for each of the 4 students. For the 4 defined tasks altogether its 30x6x4x4=2880 train procedures. If you want different amount of runs please set "runs" variable inside both "run_multiple_sin.sh", "run_multiple_flipflop.sh", and "collect_data.py" in directory "training_scripts/teacher_student/". You need to change them uniformly. Alternatively, you may set different ranks range, or run speceific task with "run_multiple_sin.sh", "run_multiple_flipflop.sh".

> Also Note that notebook can be run independently without the reproduction train - data used already located in data/teacher_student directory

### Multi-Fate inference task
To reproduce - run train script (you can also add nohup):
```
bash training_scripts/multifate_inference/run_multiple.sh > ../master_log.txt 2>&1 &
```
Next, run the notebook:

[4_MultiFate_inference.ipynb](notebooks/4_MultiFate_inference.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that notebook can be run independently without the reproduction train - data used already located in data/multifate_inference directory

### Neural recording inference tasks
> This section devided into 2 tasks -
> 1) Mante's inference task
> 2) MC-MAZE inference task

To reproduce:

1) Mante task results - code inspired by [https://github.com/adrian-valente/lowrank_inference](https://github.com/adrian-valente/lowrank_inference) - run train script (you can also add nohup):
```
bash training_scripts/mante_inference/train_mante_inference.py > ../master_log.txt 2>&1 &
```
2) MC-MAZE task results - this part is comletely based on [https://github.com/mackelab/smc_rnns](https://github.com/mackelab/smc_rnns) code. We only added Low-rank HORNN model package as an overlay to their RNN package.
First, to install their repo, and to prepare the data, run:
```
bash training_scripts/reach_inference/prepare.sh > ../master_log.txt 2>&1 &
```
Now, you can train either with or without conditioning with the train scripts (you can also add nohup):
```
bash training_scripts/reach_inference/reach_condition/run_reach_condition.sh > ../master_log.txt 2>&1 &
```
or
```
bash training_scripts/reach_inference/reach_nlb/run_reach_nlb.sh > ../master_log.txt 2>&1 &
```
respectively.

Next, run the notebook:

[5_Neural_recordings_inference.ipynb](notebooks/5_Neural_recordings_inference.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that notebook can be run independently without the reproduction train - data used already located in data/mante_inference and data/reach_inference directories
