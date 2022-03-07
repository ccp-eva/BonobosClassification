import numpy as np
import os
from argparse import ArgumentParser
import shutil
import matplotlib.pyplot as plt
import datetime
from utils import *

def main():
    parser = ArgumentParser()
    parser.add_argument(
        'database',
        default='ROI_S0',
        type=str,
        help='Video folder with category folders. ')

    log = setup_logger('my_log', 'split_database_%s.txt' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    print_and_log('Initialisation', log=log)
    args = parser.parse_args()
    

    list_categories = [f for f in os.listdir(args.database) if os.path.isdir(os.path.join(args.database,f))]
    total_samples_per_set = {'train': 0, 'validation': 0, 'test': 0}

    for category in list_categories:
        print_and_log('Processing category %s' % category, log=log)
        category_videos = os.listdir(os.path.join(args.database, category))
        N = len(category_videos)
        for dataset in total_samples_per_set.keys():
          os.makedirs(os.path.join(args.database + "_split", dataset, category), exist_ok=True)
        for idx in range(int(0.6*N)):
            shutil.copytree(os.path.join(args.database, category, category_videos[idx]), os.path.join(args.database + "_split", 'train', category, category_videos[idx]))
        for idx in range(int(0.6*N), int(0.8*N)):
            shutil.copytree(os.path.join(args.database, category, category_videos[idx]), os.path.join(args.database + "_split", 'validation', category, category_videos[idx]))
        for idx in range(int(0.8*N), N):
            shutil.copytree(os.path.join(args.database, category, category_videos[idx]), os.path.join(args.database + "_split", 'test', category, category_videos[idx]))

    print_and_log('Splitted done and saved in %s' % (args.database + "_split"), log=log)

if __name__ == '__main__':
    main()
