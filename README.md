# Beyond_Linear_Summation-Three_Body_RNNs

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

|Figure         | Reproduce |
|----------------------|------|
|1. Motivation: Three-body interactions|Created manually in Biorender.com|
|2. Neuroscience tasks and their biological gene expression counterparts|Created manually in Biorender.com|
|3. Theory validation on K-Bit Flipflop task|`1_Validation.ipynb`|
|4. Expanding solution space|`2_Solution_space.ipynb`|
|5. Mapping tasks space|`2_Solution_space.ipynb`|

## Run
### Start
To start please create a conda environment by:
```
cd Beyond_Linear_Summation-Three_Body_RNNs
conda env create -f TBRNN_env.yaml
```
(on Linux)

### Validation
First, you can reproduce validation by running train script (with nohup):
```markdown
training_scripts/teacher_student/run_multiple_flipflop.sh > master_log.txt 2>&1 &


