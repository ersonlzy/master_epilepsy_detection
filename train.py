import argparse
import torch
from engines import registered_models, registered_optimizers, registered_datasets, registered_losses, registered_schedulers, EngineBase
from engines.engine4Cla import Engine4Cla
import warnings
import datetime
import time
warnings.filterwarnings('ignore')


registered_engines = {
    "base": EngineBase,
    "cla": Engine4Cla,
}

default_type = torch.float32
torch.set_default_dtype(default_type)


def list_of_strings(arg):
	return list(map(int, arg.split(',')))

def get_parsing():
    parser = argparse.ArgumentParser('Setting Experiment', add_help=False)

    # experiment setting
    parser.add_argument('--experiment',   type=str,     default='experiment')
    parser.add_argument('--cuda',         type=str,     default='0')
    parser.add_argument('--device',       type=str,     default="cpu")
    parser.add_argument('--engine',       type=str,     default='base',       choices=list(registered_engines.keys()))
    parser.add_argument('--model',        type=str,     default='esd',        choices=list(registered_models.keys()))
    parser.add_argument('--loss',         type=str,     default="ce",         choices=list(registered_losses.keys()))
    parser.add_argument('--optim',        type=str,     default="adamw",      choices=list(registered_optimizers.keys()))
    parser.add_argument('--dataset',      type=str,     default="chbmit",     choices=list(registered_datasets.keys()))
    parser.add_argument('--scheduler',    type=str,     default="coslr",      choices=list(registered_schedulers.keys()))
    parser.add_argument('--metric_type',  type=str,     default="binary",     choices=['binary', 'multiclass'])
    parser.add_argument('--num_classes',  type=int,     default=2)
    parser.add_argument('--train_size',   type=float,   default=0.8)
    parser.add_argument('--log_step',     type=int,     default=10)
    parser.add_argument('--step',         type=int,     default=0)
    parser.add_argument('--epoch',        type=float,   default=0.0)
    parser.add_argument('--grad_cum_step',type=int,     default=1)
    parser.add_argument('--num_saves',    type=int,     default=3)
    parser.add_argument('--resume',       type=bool,    default=False)
    parser.add_argument('--checkpoint',   type=str,     default='')
    


    # trainer coeffients
    parser.add_argument('--num_epochs',   type=int,     default=100)
    parser.add_argument('--batch_size',   type=int,     default=1)
    parser.add_argument('--lr',           type=float,   default=1e-4)
    parser.add_argument('--lr0',          type=float,   default=1e-8)
    parser.add_argument('--max_norm',     type=float,   default=1.0)
    parser.add_argument('--weight_decay', type=float,   default=1e-2)
    parser.add_argument('--t0',           type=int,     default=5)
    parser.add_argument('--tmult',        type=int,     default=10)
    parser.add_argument('--init_epoch',   type=int,     default=0)
    
    # model setting
    parser.add_argument('--pred_len',     type=int,     default=96)
    parser.add_argument('--num_features', type=int,     default=23)
    parser.add_argument('--seq_len',      type=int,     default=1024)
    parser.add_argument('--diff_order',   type=int,     default=5)
    parser.add_argument('--k',            type=float,   default=0.99)
    parser.add_argument('--t',            type=int,     default=40)
    parser.add_argument('--d',            type=int,     default=5)
    parser.add_argument('--dropout',      type=float,   default=0)
    parser.add_argument('--d_model',      type=int,     default=1024)
    parser.add_argument('--channel_independent',      type=bool,     default=False)
    parser.add_argument('--task',         type=int,     default=2)

    # loss setting
    parser.add_argument('--epsilon',      type=float,   default=1e-8)
    parser.add_argument('--gamma',        type=float,   default=0.9)
    parser.add_argument('--alpha',       type=float,   default=0.5)
    
    # dataset setting
    parser.add_argument('--root_path',    type=str,     default="/Volumes/ersonlzy/datasets/chbmit/")
    parser.add_argument('--freq',         type=float,   default=256.0)
    parser.add_argument('--is_three',     type=bool,    default=False)
    parser.add_argument('--tag',          type=str,     default='train')
    parser.add_argument('--tolerance',    type=int,    default=75)
    parser.add_argument('--ts',           type=int,     default=20)
    parser.add_argument('--length',       type=int,     default=4)
    parser.add_argument('--num_workers',  type=int,     default=16)
    parser.add_argument('--shuffle',      type=bool,    default=True)
    parser.add_argument('--recut',        type=bool,    default=True)



    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Experiment and Training Script', parents=[get_parsing()])
    args = parser.parse_args()
    exp = registered_engines[args.engine](args)
    exp.run()