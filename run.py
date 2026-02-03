import datetime
import os
import time
import numpy as np
import torch, argparse, json
from config import expert_configs, dataset2path, dataset2channels
from experiments.exp_detect import Exp_Detect
from datasets.loader_provider import get_data_loader

# TODO: new
# ======================== print and log ===============================
def print_args(args, logger=None):
    """print args"""
    msg_lines = []
    msg_lines.append('=' * 80)
    msg_lines.append(f'Args Summary ({datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")})')
    msg_lines.append('-' * 80)
    for k, v in sorted(args.__dict__.items()):
        msg_lines.append(f'{k}: {v}')
    msg_lines.append('=' * 80 + '\n')

    msg = '\n'.join(msg_lines)
    print(msg)

    if logger:
        logger.write(msg+'\n')
        logger.flush()

def print_stage(title, logger=None):

    msg = f'>>>>>>>>>> {title} <<<<<<<<<<'
    print(msg)
    if logger:
        logger.write(msg+'\n')
        logger.flush()
def log_line(msg, logger=None, end='\n'):
    print(msg, end=end)
    if logger:
        logger.write(str(msg)+'\n')
        logger.flush()

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # basic training setting
    p.add_argument('--seed', type=int, default=2025, help='random seed')
    p.add_argument('--exp_name', type=str, default='msl_off_1st', help='experiment name')
    p.add_argument('--is_training', type=int, default=1, help='0: test, 1: train')
    p.add_argument('--use_gpu', type=int, default=1, help='whether to use gpu(1/0)')
    p.add_argument('--gpu', type=int, default=2, help='')
    p.add_argument('--use_multi_gpu', action='store_true', default=False, help='whether to use muti gpu')
    p.add_argument('--devices', type=str, default='0,1', help='')
    p.add_argument('--train_epochs', type=int, default=3, help='training epochs')
    p.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    p.add_argument('--use_amp', action='store_true', default=False, help='whether to use amp')
    p.add_argument('--grad_clip', type=int, default=1, help='whether to clip gradient')
    p.add_argument('--max_norm', type=float, default=1e-9, help='max norm for gradient clip')
    p.add_argument('--log_interval', type=int, default=1, help='log interval')
    p.add_argument('--patience', type=int, default=5, help='patience for early stop')
    p.add_argument('--verbose', type=int, default=1, help='verbose')
    p.add_argument('--earlystopmode', type=str, default='min', choices=['min', 'max'], help='')
    p.add_argument('--delta', type=float, default=2e-4, help='delta in early stop')
    p.add_argument('--save_every_epoch', type=bool, default=False, help='whether to save ervery epoch before es')
    p.add_argument('--monitor', type=str, default='val_loss', choices=['val_loss', 'val_metirc'], help='criterion for early stop, default:val_(tot)_loss')
    p.add_argument('--monitor_metirc', type=str, default='pa_f1', choices=['pa_f1', 'pa_rec', 'pa_pre', 'aff_rec', 'aff_pre', 'aff_f1', 'f1', 'rec', 'prec'], help='which metric to monitor for early stop when choosing val_metric')
    p.add_argument('--Iradj', type=str, default='cosine', choices=['type1', 'type2', 'type3', 'constant', 'cosine', 'card'], help='')

    # setting for model
    # basic model setting and dataset
    p.add_argument('--model', type=str, default='SEA_FGT', choices=['SEA_FGT', 'Info_FGT'],help='choose which model to experiment')
    p.add_argument('--dataset', type=str, default='MSL', choices=['MSL', 'SMD', 'SMAP', 'SWAT', 'PSM'], help='dataset name')
    p.add_argument('--data_path', type=str, default='', help='data path')
    p.add_argument('--flag', type=str, default='train', choices=['train', 'val', 'test'], help='train or test for dataloader')
    p.add_argument('--batch_size', type=int, default=32, help='batch size')
    p.add_argument('--T', type=int, default=96, help='sequence length')
    p.add_argument('--num_channels', type=int, default=0, help='number of channels')
    p.add_argument('--win_size', type=int, default=96, help='window size (equal to T)')
    p.add_argument('--step', type=int, default=24, help='step size')
    p.add_argument('--step_eq_win_size', type=int, default=0, help='step size is equal to win size? 1: yes, 0: no')

    # CCE setting
    p.add_argument('--bin_size', type=int, default=4, help='bin size for cluster')
    p.add_argument('--k_sparse', type=int, default=16, help='top k sparse for CCE')
    p.add_argument('--use_laplacian', action='store_true', default=False, help='whether to use laplacian')

    # SEA setting
    p.add_argument('--expert_configs_json', type=str, default=None, help='expert configs json file')
    p.add_argument('--topk', type=int, default=2, help='top k for SEA, default 2')
    p.add_argument('--temperature', type=float, default=1.0, help='temperature for SEA gate, default=1.0')
    p.add_argument('--capacity_factor', type=float, default=1.5, help='capacity factor for SEA gate, default=1.5')
    p.add_argument('--noise_std', type=float, default=0.2, help='noise std for SEA, default=0.1')
    p.add_argument('--prob_threshold', type=float, default=0.1, help='prob threshold for SEA , default=0.1')
    p.add_argument('--use_residual', action='store_true', default=True, help='whether to use residual in SEA')

    # transformer setting
    p.add_argument('--transformer', type=str, default='FGT', choices=['FGT', 'RGTA'], help='transformer model, default FGT')
    p.add_argument('--d_model', type=int, default=256, help='dimension in transformer')
    p.add_argument('--n_heads', type=int, default=1, help='number of heads')
    p.add_argument('--n_layers', type=int, default=3, help='number of encoder layers')
    p.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    p.add_argument('--ffn_dim', type=int, default=512, help='dimension in ffn')
    p.add_argument('--position_embedding', type=str, default='rotate', help='position embedding type')
    p.add_argument('--stat_mode', type=str, default='avg', choices=['max', 'avg'], help='statistic mode in FGA for collection, default="avg"')

    # setting for ablation
    p.add_argument('--use_cce', type=int, default=1, help='whether to use CCE')
    p.add_argument('--use_sea', type=int, default=1, help='whether to use SEA')
    p.add_argument('--use_fga', type=int, default=0, help='whether to use FGA')
    p.add_argument('--use_fgt', type=int, default=1, help='whether to use FGT')
    p.add_argument('--use_rgta', type=int, default=0, help='whether to use RGTA')

    # setting for threshold
    p.add_argument('--th_mode', type=str, default='percentile_val', choices=['percentile_val', 'percentile_train', 'spot', 'unified'], help='threshold mode')
    p.add_argument('--th_k', type=float, default=5.0, help='threshold k for percentile_val/train mode')
    p.add_argument('--th_grid', type=str, default='')
    p.add_argument('--spot_q', type=float, default=1e-5, help='threshold q for spot mode')
    p.add_argument('--spot_level', type=float, default=0.01, help='threshold level for spot mode')
    p.add_argument('--spot_scale', type=float, default=10.0, help='threshold scale for spot mode')

    # lambda
    p.add_argument('--lambda_uti', type=float, default=0.1)
    p.add_argument('--lambda_orth', type=float, default=0.0)
    p.add_argument('--lambda_info_nce', type=float, default=0.0)

    # save
    p.add_argument('--checkpoints', type=str, default='./checkpoints', help='')
    p.add_argument('--folder_path', type=str, default='./results', help='path to save results')
    p.add_argument('--save_log', type=bool, default=True, help='whether to save log')

    args = p.parse_args()

    if isinstance(args.th_grid, str) and len(args.th_grid) > 0:
        args.th_grid = [float(i) for i in args.th_grid.split(',')]
    else:
        rag = list(range(1, 101, 5))
        args.th_grid = np.array(rag)

    if args.use_cce:    args.use_cce = True
    else: args.use_cce = False
    if args.use_sea:    args.use_sea = True
    else: args.use_sea = False
    if args.use_fga:    args.use_fga = True
    else: args.use_fga = False
    if args.use_fgt:    args.use_fgt = True
    else: args.use_fgt = False


    # load expert_config json
    if args.expert_configs_json is None or args.expert_configs == '':
        args.expert_configs = expert_configs
    else:  
        args.expert_configs = json.load(open(args.expert_configs_json, 'r'))
    args.num_experts = len(args.expert_configs)

    # set for dataset
    if args.num_channels == 0:
        args.num_channels = dataset2channels[args.dataset]
    if args.data_path == '' or args.data_path is None:
        args.data_path = dataset2path[args.dataset]
    if args.T != args.win_size:
        args.win_size = args.T
    if args.step_eq_win_size == 1:
        args.step = args.T

    if args.bin_size:
        if args.bin_size > args.T // 2:
            args.bin_size = args.T // 2

    if args.k_sparse == -1:
        args.k_sparse = dataset2channels[args.dataset]

    # set for GPU
    args.use_gpu = 1 if args.use_gpu and torch.cuda.is_available() else 0
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.devices = ','.join([f'cuda:{i}' for i in device_ids])
        # args.gpu = device_ids[0]

    # set for early stop
    if args.monitor == 'val_loss':
        args.monitor_metric = None

    # save path check
    if args.exp_name == 'msl_off_1st':
        args.exp_name = f'{args.model}_{args.dataset}'
    ckpt = os.path.join(args.checkpoints, args.exp_name)
    os.makedirs(ckpt, exist_ok=True)
    resf = os.path.join(args.folder_path, args.exp_name)
    os.makedirs(resf, exist_ok=True)

    # TODO: new
    log_dir = os.path.join('logs',  f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')

    logger = open(log_path, 'w', encoding='utf-8') if args.save_log else None

    # TODO: print summary
    print_args(args, logger)

    # ready for exp
    exp = Exp_Detect(args)
    # TODO: new
    exp_name = args.exp_name
    start_all = time.time()
    
    # ok
    if args.is_training:
        # ---------- Training ----------
        print_stage(f"Start Training : {exp_name} (Dataset: {args.dataset}, Batch size: {args.batch_size})", logger)
        start_train = time.time()
        model = exp.train(setting=exp_name)
        log_line(f"Training finished in {time.time() - start_train:.2f}s.", logger)

        # ---------- Testing ----------
        print_stage(f"Start Testing : {exp_name}", logger)
        start_test = time.time()
        test_result = exp.test(setting=exp_name)
        log_line(f"Testing finished in {time.time() - start_test:.2f}s", logger)

        # 结果打印
        if isinstance(test_result, dict):
            log_line("Final Test Results:", logger)
            for k, v in test_result.items():
                log_line(f"  {k}: {v}", logger)
        
        json_path = os.path.join(resf, f"{exp_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(test_result, f)

    else:
        print_stage("Skip training, directly start testing...", logger)
        test_result = exp.test(setting=exp_name)
        
        json_path = os.path.join(args.folder_path, f"{exp_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_result, f)

    total_time = time.time() - start_all
    log_line("=" * 80, logger)
    log_line(f"All processes completed. Total time: {total_time / 60:.2f} min.", logger)
    log_line("=" * 80, logger)

    if logger:
        logger.close()