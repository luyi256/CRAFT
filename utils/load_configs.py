import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--multi_negs_batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model, note that EdgeBank is only applicable for evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--output_dim', type=int, default=172, help='dimension of the output embedding')
    parser.add_argument('--embedding_size', type=int, default=172, help='dimension of the output embedding')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--save_model_path', type=str, default='./saved_models/')
    parser.add_argument('--dataset_path', type=str, default='./data/')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works', 
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory', 
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=64, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--scale', type=float, default=12, help='scale the scores')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--is_bipartite', type=bool, default=False, help='the dataset is bipartite')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'], help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--use_edge_feat', action='store_true', default=False, help='whether to use edge feat')
    parser.add_argument('--load_pretrained', action='store_true', default=False)
    parser.add_argument('--use_node_feat', action='store_true', default=False, help='whether to use node feat')
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--use_bias', action='store_true', default=False)
    parser.add_argument('--use_act', action='store_true', default=False)
    parser.add_argument('--act_type', type=str, default='ReLU')
    parser.add_argument('--use_feat', action='store_true', default=False)
    parser.add_argument('--time_enc', action='store_true', default=False)
    parser.add_argument('--use_mixer', action='store_true', default=False)
    parser.add_argument('--use_positional', action='store_true', default=False)
    parser.add_argument('--use_neighbor', action='store_true', default=False)
    parser.add_argument('--use_layernorm', action='store_true', default=False, help='whether to use layer_norm for embedding')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--num_mixer_layers', type=int, default=2, help='number of mlp mixer layers')
    parser.add_argument('--merge', type=str, default='cat', help='model for merge two embeddings')
    parser.add_argument('--version', type=str)
    parser.add_argument('--hidden_dropout', type=float, default=0.5, help='dropout rate for hidden layers')
    parser.add_argument('--hidden_act', type=str, default='gelu', help='activation function for hidden layers')
    parser.add_argument('--loss', type=str, default='BCE', help='loss function type')
    parser.add_argument('--inner_size', type=int, default=256, help='intermediate size in feed-forward layer')
    parser.add_argument('--attn_dropout_prob', type=float, default=0.5, help='dropout rate for attention layers')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='epsilon for layer normalization')
    parser.add_argument('--initializer_range', type=float, default=0.02, help='stddev for initializing weight')
    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the training set')
    parser.add_argument('--use_pos', action='store_true', default=False, help='whether to use position embedding')
    parser.add_argument('--step', type=int, default=1, help='step for the random walk')
    parser.add_argument('--use_mean', action='store_true', default=False, help='whether to use mean in transformer')
    parser.add_argument('--output_cat_time_intervals', action='store_true', default=False, help='whether to concatenate elapsed time encoding in CRAFT')
    parser.add_argument('--output_cat_repeat_times', action='store_true', default=False, help='whether to concatenate repeat time encoding in CRAFT')
    parser.add_argument('--input_cat_time_intervals', action='store_true', default=False, help='whether to use time encoding instead of positional encoding in CRAFT')
    parser.add_argument('--num_output_layer', type=int, default=1, help='number of output layers in CRAFT')
    parser.add_argument('--collision_check', action='store_true', default=False, help='whether to check collision in train')
    parser.add_argument('--emb_dropout_prob', type=float, default=0.1, help='dropout rate for embedding')
    parser.add_argument('--use_mrr_val', action='store_true', default=False, help='whether to use mrr for validation')
    parser.add_argument('--skip_connection', action='store_true', default=False, help='whether to use skip connection in CRAFT')
    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)
        
    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['enron', 'CanParl', 'UNvote']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit', 'CanParl', 'UNtrade']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == 'JODIE':
            if args.dataset_name in ['mooc', 'USLegis']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm']:
                args.dropout = 0.3
            elif args.dataset_name in ['uci', 'UNtrade']:
                args.dropout = 0.4
            elif args.dataset_name in ['CanParl']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        elif args.model_name == 'DyRep':
            if args.dataset_name in ['mooc', 'lastfm', 'enron', 'uci', 'CanParl', 'USLegis', 'Contacts']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        else:
            assert args.model_name == 'TGN'
            if args.dataset_name in ['mooc', 'UNtrade']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm', 'CanParl']:
                args.dropout = 0.3
            elif args.dataset_name in ['enron', 'SocialEvo']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        if args.model_name in ['TGN', 'DyRep']:
            if args.dataset_name in ['CanParl'] or (args.model_name == 'TGN' and args.dataset_name == 'UNvote'):
                args.sample_neighbor_strategy = 'uniform'
            else:
                args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        if args.dataset_name in ['mooc', 'SocialEvo', 'uci', 'Flights', 'UNtrade', 'UNvote', 'Contacts']:
            args.num_neighbors = 64
        elif args.dataset_name in ['lastfm', 'CanParl']:
            args.num_neighbors = 128
        else:
            args.num_neighbors = 32
        if args.dataset_name in ['CanParl']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'EdgeBank':
        if args.negative_sample_strategy == 'random':
            if args.dataset_name in ['wikipedia', 'reddit', 'uci', 'Flights']:
                args.edge_bank_memory_mode = 'unlimited_memory'
            elif args.dataset_name in ['mooc', 'lastfm', 'enron', 'CanParl', 'USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['UNtrade', 'UNvote', 'Contacts']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name == 'SocialEvo'
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
        elif args.negative_sample_strategy == 'historical':
            if args.dataset_name in ['uci', 'CanParl', 'USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['mooc', 'lastfm', 'enron', 'UNtrade', 'UNvote', 'Contacts']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name in [
                    'wikipedia', 'reddit', 'SocialEvo', 'Flights']
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
        else:
            assert args.negative_sample_strategy == 'inductive'
            if args.dataset_name in ['USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['uci', 'UNvote']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron',
                                             'SocialEvo', 'Flights', 'CanParl', 'UNtrade', 'Contacts']
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['SocialEvo', 'uci', 'UNtrade', 'UNvote', 'Contacts']:
            args.dropout = 0.0
        elif args.dataset_name in ['CanParl']:
            args.dropout = 0.2
        elif args.dataset_name in ['USLegis']:
            args.dropout = 0.3
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit', 'CanParl', 'USLegis', 'UNtrade', 'UNvote']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['wikipedia']:
            args.num_neighbors = 30
        elif args.dataset_name in ['reddit', 'lastfm']:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        if args.dataset_name in ['wikipedia', 'reddit', 'enron']:
            args.dropout = 0.5
        elif args.dataset_name in ['mooc', 'uci', 'USLegis']:
            args.dropout = 0.4
        elif args.dataset_name in ['lastfm', 'UNvote']:
            args.dropout = 0.0
        elif args.dataset_name in ['SocialEvo']:
            args.dropout = 0.3
        elif args.dataset_name in ['Flights', 'CanParl']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ['CanParl', 'UNtrade', 'UNvote']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['DyGFormer']:
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ['mooc', 'enron', 'Flights', 'USLegis', 'UNtrade']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ['CanParl']:
            args.max_input_sequence_length = 2048
            args.patch_size = 64
        elif args.dataset_name in ['UNvote']:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit', 'UNvote']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron', 'USLegis', 'UNtrade', 'Contacts']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")

