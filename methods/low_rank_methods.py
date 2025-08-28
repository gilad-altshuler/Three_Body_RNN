import torch
from torch import nn
from tasks.K_Bit_Flipflop_task import accuracy
from methods.models import *


def TCA_method(model,input,target,start_rank=1,end_rank=6,W0=None):
    """
    Apply low rank truncation according to Tensor Component Analysis (TCA) method.
    :param model: Full rank TBRNN to truncate its connectivity.
    :param input: Inputs to evaluate the truncated model.
    :param target: Targets to evaluate the truncated model.
    :param start_rank: rank to start the truncation from.
    :param end_rank: rank to stop the truncation method.
    :param W0: (optional) should be pass if try truncate delta W instead of W tensor.
    :return: accuracy array for each rank from start_rank to end_rank.
    """
    import tensorly as tl
    from tensorly.decomposition import parafac
    DEVICE = next(model.parameters()).device

    accs = []
    newModel = model.clone()
    for rank in range(start_rank,end_rank+1):
      # calculate TCA truncated accuracy
      twt = model.w_hh.cpu().detach().clone().numpy()
      if W0 is not None:
          twt = twt - W0.cpu().detach().clone().numpy()
      factors = parafac(twt, rank=rank)
      twt = torch.tensor(tl.cp_to_tensor(factors),device=DEVICE)
      if W0 is not None:
          twt = twt + W0.detach().clone().to(DEVICE)
      newModel.w_hh = torch.nn.Parameter(twt)
      accs.append(accuracy(newModel(input,None)[0],target))
    return np.array(accs)


def TT_method(model,input,target,start_rank=1,end_rank=6,W0=None):
    """
    Apply low rank truncation according to Tensor Train (TT) method.
    :param model: Full rank TBRNN to truncate its connectivity.
    :param input: Inputs to evaluate the truncated model.
    :param target: Targets to evaluate the truncated model.
    :param start_rank: rank to start the truncation from.
    :param end_rank: rank to stop the truncation method.
    :param W0: (optional) should be pass if try truncate delta W instead of W tensor.
    :return: accuracy array for each rank from start_rank to end_rank.
    """
    import tensorly as tl
    from tensorly.decomposition import TensorTrain
    DEVICE = next(model.parameters()).device

    accs = []
    newModel = model.clone()
    for rank in range(start_rank,end_rank+1):
        twt = model.w_hh.cpu().detach().clone().numpy()
        if W0 is not None:
            twt = twt - W0.cpu().detach().clone().numpy()
        tensor = tl.tensor(twt)
        tt = TensorTrain(rank=[1,rank,newModel.hidden_dim,1])
        factors = tt.fit_transform(tensor)
        twt = torch.tensor(tl.tt_to_tensor(factors),device=DEVICE)
        if W0 is not None:
            twt = twt + W0.detach().clone().to(DEVICE)
        newModel.w_hh = torch.nn.Parameter(twt)
        accs.append(accuracy(newModel(input,None)[0],target))
    return np.array(accs)


def sliceTCA_method(model,input,target,start_rank=1,end_rank=6,W0=None):
    """
    Apply low rank truncation according to sliceTCA method (see paper - Dimensionality reduction beyond neural
    subspaces with slice tensor component analysis.
    Nature Neuroscience https://www.nature.com/articles/s41593-024-01626-2).
    :param model: Full rank TBRNN to truncate its connectivity.
    :param input: Inputs to evaluate the truncated model.
    :param target: Targets to evaluate the truncated model.
    :param start_rank: rank to start the truncation from.
    :param end_rank: rank to stop the truncation method.
    :param W0: (optional) should be pass if try truncate delta W instead of W tensor.
    :return: accuracy array for each rank from start_rank to end_rank.
    """
    import slicetca
    DEVICE = next(model.parameters()).device

    accs = []
    newModel = model.clone()

    twt = newModel.w_hh.detach().clone()
    if W0 is not None:
        twt = twt - W0.detach().clone()

    for rank in range(start_rank, end_rank+1):
        _, mod = slicetca.decompose(twt / twt.std(), number_components=(rank, 0, 0))
        # For a not positive decomposition, we apply uniqueness constraints
        mod = slicetca.invariance(mod)
        reconstruction_full = mod.construct().detach() * twt.std()
        if W0 is not None:
            reconstruction_full = (reconstruction_full + W0.detach().clone()).to(DEVICE)
        newModel.w_hh = torch.nn.Parameter(reconstruction_full)
        accs.append(accuracy(newModel(input,None)[0],target))
        torch.cuda.empty_cache()
    return np.array(accs)

def LINT_method(teacher,student_class,input,target,start_rank=1,end_rank=6,epochs=10000,batch_size = 128,
                lr = [1e-03,1e-03],to_save="",return_accs=True,rates=False,sched_epochs=10000):
    """
    Apply low rank truncation according to Low Rank Inference (LINT) method.
    :param teacher: Full rank model to perform the low rank approximation on.
    :param student_class: Low rank Inference (LINT) approximation model type.
    :param input: Inputs to evaluate the truncated model.
    :param target: Targets to evaluate the truncated model.
    :param start_rank: rank to start the truncation from.
    :param end_rank: rank to stop the truncation method.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param lr: Learning rate for the optimizer.
    :param to_save: Path to save the trained model.
    :param return_accs: If True, return accuracy for each rank, otherwise return None.
    :param rates: If True, apply output nonlinearity to the model trajectory.
    :return: accuracy array for each rank from start_rank to end_rank if return_accs is True, otherwise None.
    """

    # MSE loss and Adam optimizer with a learning rate
    DEVICE = next(teacher.parameters()).device

    criterion = nn.MSELoss().to(DEVICE)
    teacher_hidden = teacher(input, None)[2].detach().clone()  # get teacher hidden state
    if rates:
        teacher_hidden = teacher.output_nonlinearity(teacher_hidden)
    accs = np.array([])
    for rank in range(start_rank, end_rank + 1):

        model_path = f"r_{rank}_{get_model_str(student_class)}_on_{get_model_str(type(teacher))}.pth"
        if to_save and (to_save / model_path).exists():
            print(f"Model {model_path} already exists")
            continue

        # instantiate an low rank RNN
        output_nonlinearity = teacher.output_nonlinearity if rates else (lambda x: x)
        lr_student = student_class(teacher.input_size, teacher.output_size, teacher.hidden_dim, rank, teacher.nonlinearity,
                              output_nonlinearity,teacher.task, teacher.mode, teacher.form, teacher.noise_std, teacher.tau,
                              teacher.Win_bias, teacher.Wout_bias,w_out=torch.nn.Identity().to(DEVICE),hard_orth=teacher.hard_orth).to(DEVICE)
        end_factor = lr[1] / lr[0]
        optimizer = torch.optim.Adam(lr_student.parameters(), lr=lr[0])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=end_factor,total_iters=sched_epochs)
        dataset = (input,teacher_hidden)
        _ = train(lr_student,dataset,epochs,optimizer,criterion,
                  scheduler=scheduler,batch_size=batch_size,clip_gradient=None,keep_best=True,plot=False)
        
        lr_student.output_nonlinearity = teacher.output_nonlinearity
        lr_student.w_out = copy.deepcopy(teacher.w_out)
        
        if return_accs:
            accs = np.append(accs, accuracy(lr_student(input,None)[0],target))
        else:
            accs = None

        if to_save:
            torch.save(lr_student.state_dict(), to_save / model_path)

    return accs

