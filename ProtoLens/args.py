import os
import argparse
from datetime import datetime
import json
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-mp', '--master_port', type=str, default='123314', help='Set the master port.')
parser.add_argument('-pn', '--prototype_num', type=int, default=20)
parser.add_argument('-setting', '--setting', type=str, default="fine-tune")
parser.add_argument('-e', '--epoch', type=int, default=25, help='Set the total epoch.')
# parser.add_argument('-l', '--layer', type=int, default=12, help='embedding layer')
parser.add_argument('-i', '--device_ids', type=str, default="0", help='Set the device (GPU ids). Split by @. E.g., 0@2@3.')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('-rc', '--round_count', type=int, default=0, help='Count the round of experiments.')
parser.add_argument('-ma', '--master_address', type=str, default='127.0.0.1', help='Set the master address.')
parser.add_argument('-li', '--log_iter', type=int, default=18, help='The number of iterations (batches) to log once.')
parser.add_argument('-gau', '--gaussian_num', type=int, default=6, help='The number of Gaussian components')
parser.add_argument('-ws', '--window_size', type=int, default=5, help='sliding window size')
# --------------------------------------------------------------------------------------
parser.add_argument('-d', '--data_set', type=str, default='IMDB',
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument('-bb', '--bert_model_name', type=str, default='all-mpnet-base-v2') # distilbert-base-uncased-finetuned-sst-2-english  multi-qa-distilbert-cos-v1 all-MiniLM-L6-v2
parser.add_argument('-hidden', '--hidden_dim', type=int, default=768)
parser.add_argument('-ml','--max_length', type=int, default=512)
parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Set the batch size.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Set the initial learning rate.')
# ---------------------------------------------------------------------------------------
# parser.add_argument('--save_best', action="store_true",default=True, help='Save the model with best performance on the validation set.')
parser.add_argument('-seed', '--r_seed', type = int, default=42)
# ---------------------------------------------------------------------------------------
# pnfrl_args, unknown = parser.parse_known_args()
pnfrl_args = parser.parse_args()
pnfrl_args.folder_name = '_{}_{}_{}_gNum_{}_ws_{}_e_{}_pNum_{}_lr{}'.format(
    pnfrl_args.data_set, pnfrl_args.setting, pnfrl_args.bert_model_name, pnfrl_args.gaussian_num, pnfrl_args.window_size, pnfrl_args.epoch, pnfrl_args.prototype_num,  pnfrl_args.learning_rate)
pnfrl_args.base_folder = "Datasets"
if pnfrl_args.data_set == "IMDB":
    pnfrl_args.dataset_path = "/Datasets/IMDB/IMDB Dataset.csv"
else: 
    pnfrl_args.dataset_path = " "
pnfrl_args.num_classes = 2
if not os.path.exists('log_folder'):
    os.mkdir('log_folder')
# pnfrl_args.folder_name = pnfrl_args.folder_name + '_L' + pnfrl_args.structure
pnfrl_args.set_folder_path = os.path.join('log_folder', pnfrl_args.data_set)
if not os.path.exists(pnfrl_args.set_folder_path):
    os.mkdir(pnfrl_args.set_folder_path)
pnfrl_args.folder_path = os.path.join(pnfrl_args.set_folder_path, pnfrl_args.folder_name)
if not os.path.exists(pnfrl_args.folder_path):
    os.mkdir(pnfrl_args.folder_path)
pnfrl_args.model_path = os.path.join(pnfrl_args.folder_path, 'model.pth')
pnfrl_args.log = os.path.join(pnfrl_args.folder_path, 'log.txt')
pnfrl_args.device_ids = list(map(int, pnfrl_args.device_ids.strip().split('@')))
pnfrl_args.gpus = len(pnfrl_args.device_ids) 
pnfrl_args.nodes = 1
pnfrl_args.world_size = (pnfrl_args.gpus * pnfrl_args.nodes )
pnfrl_args.batch_size = int(pnfrl_args.batch_size)#int(pnfrl_args.batch_size / pnfrl_args.gpus)
