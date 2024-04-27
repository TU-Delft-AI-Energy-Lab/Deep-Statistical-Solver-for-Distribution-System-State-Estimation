from functools import partial
from argparse import ArgumentParser, Namespace
import os
import yaml
import warnings

def str2tuple(str, dtype=int):
    " str: '[1,2,3]' "
    str = str.replace(" ", "")
    if str[0] == '[' and str[-1] == ']' or str[0] == '(' and str[-1] == ')':
        str = str[1:-1]
    return tuple(map(dtype, str.split(',')))

str2float_tuple = partial(str2tuple, dtype=float)
str2int_tuple = partial(str2tuple, dtype=int)
str2str_tuple = partial(str2tuple, dtype=str)

def str2bool(str):
    if isinstance(str, bool):
        return str
    if str.upper() in {'TRUE', 'T', '1'}:
        return True
    elif str.upper() in {'FALSE', 'F', '0'}:
        return False
    else:
        raise ValueError('Invalid boolean value')

def argument_parser() -> Namespace:
    "parse all the arguments here"
    config_parser = ArgumentParser(
        prog='HeatoDiff',
        description='parse yaml configs',
        add_help=False)
    config_parser.add_argument('--config', '--configs', default='configs/unet.yaml', type=str)
    
    parser = ArgumentParser(
        prog='HeatoDiff',
        description='train neural network for heat pump consumption profile generation'
    )
    
    # General Parameters
    parser.add_argument('--exp_id', default='0.0.0', type=str, help='experiment id')
    
    # Data Parameters
    # parser.add_argument('--data_path', default='/home/nlin/data/volume_2/heat_profile_dataset/2019_data_15min.hdf5', type=str, help='path to data')
    parser.add_argument('--dataset', default='not specified', type=str, help='dataset name, should be one of ["wpuq", "lcl_electricity"]')
    parser.add_argument('--data_root', default='/home/nlin/data/volume_2/heat_profile_dataset', type=str, help='root directory of data')
    parser.add_argument('--data_case', default='2019_data_15min.hdf5', type=str, help='case name of data') # NOTE: this is not used
    parser.add_argument('--target_labels', default=None, type=str2str_tuple, help='target labels of data, e.g. ["year","season"]')
    parser.add_argument('--resolution', default='15min', type=str, help='resolution of data, one of [10s, 1min, 15min, 30min, 1h]')
    parser.add_argument('--cossmic_dataset_names', default=[], type=str2str_tuple, help='dataset names of CoSSMic, e.g. ["grid_import_residential"]')
    parser.add_argument('--season', default='winter', type=str, help='season of data') # NOTE: this is not used
    parser.add_argument('--train_season', default='winter', type=str, help='season of training data, winter/spring/.../whole_year')
    parser.add_argument('--val_season', default='winter', type=str, help='season of validation data, winter/spring/.../whole_year')
    parser.add_argument('--val_area', default='all', type=str, help='area of validation data, industrial/residential/public')
    parser.add_argument('--load_data', default=True, type=str2bool, help='whether to load processed data')
    parser.add_argument('--normalize_data', default=True, type=str2bool, help='whether to normalize data')
    parser.add_argument('--pit_data', default=False, type=str2bool, help='whether to PIT data')
    parser.add_argument('--shuffle_data', default=True, type=str2bool, help='whether to shuffle data')
    parser.add_argument('--vectorize_data', default=False, type=str2bool, help='whether to vectorize data')
    parser.add_argument('--style_vectorize', default=False, type=str, help='whether to vectorize data with style, \
        should be one of ["chronological", "stft"]')
    parser.add_argument('--vectorize_window_size', default=3, type=int, help='window size for vectorization')
    # NOTE train/val/test ratio is not used anymore. 
    parser.add_argument('--train_ratio', default=0.7, type=float, help='ratio of training data')
    parser.add_argument('--val_ratio', default=0.15, type=float, help='ratio of validation data')
    parser.add_argument('--test_ratio', default=0.15, type=float, help='ratio of testing data')
    
    # Network Parameters
    parser.add_argument('--model_class', default='unet', type=str, help='model class, should be one of ["unet", "gpt2", "mlp]')
    parser.add_argument('--conditioning', default=False, type=str2bool, help='whether to use conditioning')
    parser.add_argument('--cond_dropout', default=0.1, type=float, help='dropout rate for conditioning')
    parser.add_argument('--dim_base', default=128, type=int, help='base dimension for convs')
    parser.add_argument('--dim_mult', default=(1, 2, 4, 8), type=str2int_tuple, help='dimension multiplier for convs')
    
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--num_attn_head', default=4, type=int, help='number of attention heads')
    # parser.add_argument('--type_transformer', default='gpt2', type=str, help='type of transformer, \
    #     should be one of ["gpt2", "transformer"]')
    parser.add_argument('--num_encoder_layer', default=6, type=int, help='number of encoder layers')
    parser.add_argument('--num_decoder_layer', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dimension of feedforward layer')
    
    # Diffusion Parameters
    parser.add_argument('--num_diffusion_step', default=1000, type=int, help='number of diffusion steps')
    parser.add_argument('--num_sampling_step', default=-1, type=int, help='number of sampling steps')
    parser.add_argument('--diffusion_objective', default='pred_v', type=str, help='objective for diffusion, \
        should be one of ["pred_v", "pred_noise", "pred_x0"]')
    parser.add_argument('--learn_variance', default=False, type=str2bool, help='whether to learn variance')
    parser.add_argument('--sigma_small', default=True, type=str2bool, help='whether to use small sigma if not learned')
    parser.add_argument('--beta_schedule_type', default='cosine', type=str, help='type of beta schedule, \
        should be one of ["cosine", "linear"]')
    parser.add_argument('--ddim_sampling_eta', default=0., type=float, help='eta parameter of the ddim sampling')
    parser.add_argument('--cfg_scale', default=1., type=float, help='classfier-free guidance scale, default=1. (no guidance)')
    
    # Training Parameters
    #   batch size and optimizer
    parser.add_argument('--mse_loss', default=True, type=str2bool, help='whether to use MSE loss or KL loss')
    parser.add_argument('--rescale_learned_variance', default=True, type=str2bool, help='whether to rescale loss of learned variance')
    parser.add_argument('--only_central', default=False, type=str2bool, help='only use the central channel of output for loss.')
    parser.add_argument('--train_batch_size', default=480, type=int, help='batch size')
    parser.add_argument('--train_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--adam_betas', default=(0.9,0.999), type=str2float_tuple, help='betas for adam optimizer')
    #   trainer and ema
    parser.add_argument('--gradient_accumulate_every', default=2, type=int, help='gradient accumulation steps')
    parser.add_argument('--ema_update_every', default=10, type=int, help='ema update steps')
    parser.add_argument('--ema_decay', default=0.995, type=float, help='ema decay')
    parser.add_argument('--amp', default=False, type=str2bool, help='whether to use amp')
    parser.add_argument('--mixed_precision_type', default='fp16', type=str, help='mixed precision type, should be one of ["fp16", "fp32"]')
    parser.add_argument('--split_batches', default=True, type=str2bool, help='whether to split batches for accelerator')
    #   train and logging steps
    parser.add_argument('--num_train_step', default=5000, type=int, help='number of training steps')
    parser.add_argument('--save_and_sample_every', default=500, type=int, help='save and sample every n steps')
    parser.add_argument('--val_every', default=1000, type=int, help='validate every n steps. save_and_sample_every must be a multiple of val_every')
    parser.add_argument('--num_sample', default=25, type=int, help='number of samples to generate every n steps')
    parser.add_argument('--log_wandb', default=True, type=str2bool, help='whether to log to wandb')
    
    # Sample and test Parameters
    parser.add_argument('--val_batch_size', default=None, type=int, help='batch size for validation/testing')
    parser.add_argument('--load_runid', default=None, type=str, help='run id to load')
    parser.add_argument('--load_milestone', default=10, type=int, help='milestone to load')
    parser.add_argument('--resume', default=False, action='store_true', help='whether to resume training from a milestone')
    parser.add_argument('--freeze_layers', default=False, action='store_true', help='whether to freeze layers of transformer')
    
    # Parsing arguments
    #   Step 0: Parse args in config yaml
    args, left_argv = config_parser.parse_known_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_yaml_dict = yaml.safe_load(f)
        
        yaml_argv = []
        for key, value in config_yaml_dict.items():
            yaml_argv.append('--' + key)
            yaml_argv.append(str(value))
        parser.parse_known_args(yaml_argv, namespace=args) # write config yaml values to args
    #   Step 1: Parse args in command line
    parser.parse_args(left_argv, namespace=args)
    
    # Post processing
    if args.dataset == 'lcl_electricity':
        args.resolution = '30min' # fixed. 
    if args.dataset == 'cossmic':
        assert len(args.cossmic_dataset_names) > 0, 'cossmic_dataset_names is empty.'
    if args.val_batch_size is None:
        args.val_batch_size = args.train_batch_size
    if args.dataset == 'not specified':
        raise ValueError('dataset is not specified.')
    if not args.shuffle_data:
        warnings.warn('shuffle_data is False.')
    #   train/val/test ratio
    # NOTE not used anymore
    _sum = args.train_ratio + args.val_ratio + args.test_ratio
    args.train_ratio /= _sum
    args.val_ratio /= _sum
    args.test_ratio /= _sum
    
    #   num sampling step
    if args.num_sampling_step == -1:
        args.num_sampling_step = args.num_diffusion_step
    
    return args