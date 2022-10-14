import argparse
import torch
import os


parser = argparse.ArgumentParser(description='HGWaveNet')

parser.add_argument('--dataset', type=str, default='dblp', help='dataset name')
parser.add_argument('--data_pt_path', type=str, default='./data/', help='parent path of dataset')
parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--nhid', type=int, default=16, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=16, help='dim of output embedding')

parser.add_argument('--max_epoch', type=int, default=500, help='number of epochs to train.')
parser.add_argument('--min_epoch', type=int, default=50, help='min epoch')
parser.add_argument('--test_length', type=int, default=3, help='length for test, default:3')
parser.add_argument('--device', type=int, default=0, help='gpu id, -1 for cpu')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--repeat', type=int, default=5, help='running times')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic model.')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--log_interval', type=int, default=1, help='log interval, default: 20,[20,40,...]')
parser.add_argument('--pre_defined_feature', default=None, help='pre-defined node feature')
parser.add_argument('--save_embeddings', type=int, default=0, help='save or not, default:0')
parser.add_argument('--output_pt_path', type=str, default='./output/', help='parent path of output')
parser.add_argument('--debug_mode', type=int, default=0, help='debug_mode, 0: normal running; 1: debugging mode')
parser.add_argument('--use_riemannian_adam', type=bool, default=True,
                    help='use riemannian adam or original adam as optimizer')

parser.add_argument('--model', type=str, default='HGWaveNet', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall', help='hyperbolic model')
parser.add_argument('--use_hyperdecoder', type=bool, default=True, help='use hyperbolic decoder or not')
parser.add_argument('--spatial_dilated_factors', type=list, default=[1, 2],
                    help='dilated factor for dilated spatial convolution')
parser.add_argument('--casual_conv_depth', type=int, default=3, help='number of temporal casual convolution layers')
parser.add_argument('--casual_conv_kernel_size', type=int, default=2,
                    help='temporal casual convolution kernel size')
parser.add_argument('--eps', type=float, default=1e-15, help='eps')
parser.add_argument('--bias', type=bool, default=True, help='use bias or not')
parser.add_argument('--trainable_feat', type=bool, default=True,
                    help='using trainable feat or one-hot feat, default: trainable feat')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (1 - keep probability).')
parser.add_argument('--heads', type=int, default=1, help='attention heads.')
parser.add_argument('--curvature', type=float, default=1, help='curvature value')
parser.add_argument('--trainable_curvature', type=bool, default=False, help='trainable curvature or not')
parser.add_argument('--aggregation', type=str, default='deg', help='aggregation method: [deg, att]')

args = parser.parse_args()

if args.device >= 0 and torch.cuda.is_available():
    args.device = torch.device('cuda:{}'.format(args.device))
else:
    args.device = torch.device('cpu')
print('Using device {} to train the model ...'.format(args.device))

args.output_path = os.path.join(args.output_pt_path, args.dataset)
if not os.path.isdir(args.output_path):
    os.makedirs(args.output_path)
args.log_file = os.path.join(args.output_path, '{}.log'.format(args.model))
args.emb_file = os.path.join(args.output_path, '{}.emb'.format(args.model))
