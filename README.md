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


## Regenerate paper figures

|Figure / Table         | Reproduce |
|----------------------|------|
|Fig 1. Motivation: Three-body interactions|Created manually in Biorender.com|
|Fig 2. Neuroscience tasks and their biological gene expression counterparts|Created manually in Biorender.com|
|Fig 3. Theory validation on K-Bit Flipflop task|`1_Low_rank_TBRNN_validation.ipynb`|
|Fig 4. Expanding solution space|`2_Solution_space.ipynb`|
|Fig 5. Mapping tasks space|Created manually in Biorender.com|
|Table 1. Teacher-student inference results on K-bit Flip-Flop and sine wave synthetic tasks|`3_Teacher_Student.ipynb`|
|Table 2. Model CKA scores performance on 30-MultiFate inference task|`4_MultiFate inference.ipynb`|

## Run
### Start
To start please create a conda environment by:
```
cd Three_Body_RNN
conda env create -f TBRNN_env.yaml
conda activate TBRNN_env
```
(on Linux)

### Low-rank TBRNN validation
To reproduce the low-rank and theory validation data - run train script with nohup:
```
training_scripts/validation/run_multiple.sh > ../master_log.txt 2>&1 &
```
Next, run the notebook:

[notebooks/1_Low_rank_TBRNN_validation.ipynb](notebooks/1_Low_rank_TBRNN_validation.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that notebook can be run independently without the reproduction train - data used already located in data/validation directory

### Expanding solution space
To reproduce - run train script with nohup:
```
training_scripts/solution_space/run_multiple.sh > ../master_log.txt 2>&1 &
```
Next, run the notebook:

[notebooks/2_Solution_space.ipynb](notebooks/2_Solution_space.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that notebook can be run independently without the reproduction train - data used already located in data/solution_space directory

### Teacher-student setup on synthetic neuroscience data
To reproduce - run train script with nohup:
```
training_scripts/teacher_student/run_multiple_tasks.sh > ../master_log.txt 2>&1 &
```
Next, run the notebook:

[notebooks/3_Teacher_Student.ipynb](notebooks/3_Teacher_Student.ipynb)
*(May be run via **google colab** or **linux terminal**)*
> Note that notebook can be run independently without the reproduction train - data used already located in data/teacher_student directory
