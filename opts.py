# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
import multiprocessing
import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--dataset', default='minikinetics', type=str)
parser.add_argument('--modality', default='RGB', type=str)
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")

# ========================= Custom Configs ==========================
parser.add_argument('--two_stream', default=True, action="store_false", help='add frequency stream')
parser.add_argument('--warm_up_epoch', default=0, type=int, help="0 when 1stream")

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='scratch')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=196, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=10, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=multiprocessing.cpu_count() // 3 * 4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
