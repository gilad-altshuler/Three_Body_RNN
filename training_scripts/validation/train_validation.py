import sys, os
import importlib
from pathlib import Path
sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

import torch
from torch import nn
from low_rank_procedures import TCA_method, TT_method, sliceTCA_method, LINT_method
from Models import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_ROOT = Path(__file__).absolute().parent.parent.parent.parent / "runs" / "low_rank_procedures"

def run_methods(Model_class,lr_class,run_name,ranks=[1,6],data_size=128,
                T=100,input_size=3,output_size=3,hidden_dim=30,epochs=10000,
                lint_epochs=30000,lr=[5e-4,1e-4],lint_lr=[1e-3,1e-3]):

    run_dir = RUN_ROOT / run_name
    if not os.path.isdir(run_dir):
        run_dir.mkdir(parents=True)

    end_factor = lr[1]/lr[0]

    criterion = nn.MSELoss().to(DEVICE)

    generate_data = getattr("tasks."+importlib.import_module(run_name.split('/')[0]), 'generate_data')
    input, target = generate_data(data_size,T,input_size)

    while True:
        # define teacher
        teacher = Model_class(input_size, output_size, hidden_dim, mode='disc',
                        form='voltage',task="FF", Win_bias=True, Wout_bias=True).to(DEVICE)

        optimizer = torch.optim.Adam(teacher.parameters(), lr=lr[0])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=end_factor,total_iters=epochs)

        _ = train(teacher,input,target,epochs,optimizer,criterion,
                    scheduler=scheduler,mask_train=None,batch_size=data_size,
                    hidden=None,clip_gradient=None,keep_best=True,plot=False)

        eps = 0.05

        accuracy = getattr("tasks."+importlib.import_module(run_name.split('/')[0]), 'accuracy')
        acc = accuracy(teacher(input,None)[0],target)
        if acc < (1-eps):
            print(f"Teacher model not trained, Accuracy: {acc}... trying again.")
        else:
            print(f"Teacher model trained. Accuracy: {acc}")
            torch.save(teacher.state_dict(), run_dir / "teacher_model.pth")
            break

    # TCA method
    tca = TCA_method(teacher, input, target, ranks[0], ranks[1])
    print("TCA accuracies:", tca)
    # TT method
    tt = TT_method(teacher, input, target, ranks[0], ranks[1])
    print("TT accuracies:", tt)
    # sliceTCA method
    slice_tca = sliceTCA_method(teacher, input, target, ranks[0], ranks[1])
    print("sliceTCA accuracies:", slice_tca)
    # LINT method
    lint = LINT_method(teacher,lr_class,input,target,start_rank=ranks[0],end_rank=ranks[1],
                       epochs=lint_epochs,batch_size = data_size,lr = lint_lr,to_save=run_dir,return_accs=True,rates=False)
    print("LINT accuracies:", lint)
    # save results
    np.save(run_dir / "tca.npy", tca)
    np.save(run_dir / "tt.npy", tt)
    np.save(run_dir / "slice_tca.npy", slice_tca)
    np.save(run_dir / "lint.npy", lint) 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINT")
    parser.add_argument("--model", "-m", type=str, default="TBRNN",
                        choices=["TBRNN", "Full_Rank_RNN", "Low_Rank_TBRNN", "Low_Rank_RNN"],
                        help="Model class to use for the low rank approximation.")
    parser.add_argument("--lr_class", "-lc", type=str, default="Low_Rank_TBRNN",
                        choices=["Low_Rank_RNN", "Low_Rank_TBRNN","Low_Rank_HORNN"],
                        help="Low rank class to use for the student model.")
    parser.add_argument("--ranks", "-ranks", type=int, nargs=2, default=[1, 6])
    parser.add_argument("--run_name", "-r", type=str)
    parser.add_argument("--data_size", "-d", type=int, default=128)
    parser.add_argument("--T", "-T", type=int, default=100)
    parser.add_argument("--input_size", "-is", type=int, default=3)
    parser.add_argument("--output_size", "-os", type=int, default=3)
    parser.add_argument("--hidden_dim", "-hd", type=int, default=30)
    parser.add_argument("--epochs", "-e", type=int, default=10000)
    parser.add_argument("--lint_epochs", "-le", type=int, default=30000)
    parser.add_argument("--lr", "-lr", type=float, nargs=2, default=[5e-4, 1e-4])
    parser.add_argument("--lint_lr", "-llr", type=float, nargs=2, default=[1e-3, 1e-3])
    args, extras = parser.parse_known_intermixed_args()


    run_methods(globals()[args.model],globals()[args.lr_class],args.run_name,
                args.ranks,args.data_size,T=args.T,input_size=args.input_size,output_size=args.output_size,
                hidden_dim=args.hidden_dim,epochs=args.epochs,lint_epochs=args.lint_epochs,lr=args.lr,lint_lr=args.lint_lr)
