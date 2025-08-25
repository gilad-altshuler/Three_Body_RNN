# Beyond Linear Summation - Three Body RNNs

## About
Code repository accompaning the paper "Beyond linear summation: Three-Body RNNs for modeling complex neural and biological systems"

<img width="2250" height="1500" alt="Three Body interactions" src="https://github.com/user-attachments/assets/0a406309-1f0f-4786-9e3c-6a9fe0a1ce11" />
Fig 1. Motivation: Three-body interactions. (a) Neuromodulatory axon operate
as a third-body in the synapse. (b) Incorporating glia cells into the network induce
nonlinear summation. (c) Dendritic nonlinearities, where inputs on the same branch
gate each other. (d) Gene expression networks; monomers that can dimerize and serve
as transcription factors for each other, leading to full-rank three-body interaction (TF
in panel = Transcription Factor). (e) Traditional neural network with linear
pre-synaptic summation (left) while biophysics is more complex and input couplings
present (right). 


## Reproduce paper figures

|Figure / Table         | Reproduce + Link to instructions |
|----------------------|------|
|Fig 1. Motivation: Three-body interactions|Created manually in Biorender.com|
|Fig 2. Neuroscience tasks and their biological gene expression counterparts|Created manually in Biorender.com|
|Fig 3. Theory validation on K-Bit Flipflop task|[`1_Low_rank_TBRNN_validation.ipynb`](#Low-rank-TBRNN-validation)|
|Fig 4. Expanding solution space|[`2_Solution_space.ipynb`](#Expanding-solution-space)|
|Fig 5. Mapping tasks space|Created manually in Biorender.com|
|Table 1. Teacher-student inference results on K-bit Flip-Flop and sine wave synthetic tasks|[`3_Teacher_Student.ipynb`](#Teacher-student-setup-on-synthetic-neuroscience-data)|
|Table 2. Model CKA scores performance on 30-MultiFate inference task|[`4_MultiFate inference.ipynb`](#Multi-Fate-inference-task)|

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

### Monkey neural trajectory inference tasks
> This section devided into 2 tasks -
> 1) Mante's inference task
> 2) Macaque inference task
To reproduce:
1) Mante task results - code inspired by [https://github.com/adrian-valente/lowrank_inference](https://github.com/adrian-valente/lowrank_inference) - run train script (you can also add nohup):
```
bash training_scripts/mante_inference/train_mante_inference.py > ../master_log.txt 2>&1 &
```
2) Macaque task results - this part is comletely based on [https://github.com/mackelab/smc_rnns](https://github.com/mackelab/smc_rnns) code. We only added Low-rank HORNN model package as an overlay to their RNN package.
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

[5_Neural_Trajectory_inference.ipynb](notebooks/5_Neural_Trajectory_inference.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that notebook can be run independently without the reproduction train - data used already located in data/mante_inference and data/reach_inference directories
