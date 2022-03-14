import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import os
import gc
import copy
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from utils import *
from argparse import ArgumentParser

def visualize_database(inputs, labels, databse_name, save_path):
    
    fig = plt.figure()
    fig.suptitle('Database samples', fontsize=8)

    inputs = inputs.to("cpu")
    labels = labels.to("cpu")

    for j in range(inputs.size()[0]):
        ax = plt.subplot(inputs.size()[0]//3, 3, j+1)
        ax.axis('off')                
        ax.set_title('%s-%s (%.2f/%.2f)' % (class_names[labels[j]], class_names[preds[j]], outputs[j][preds[j]], output_normalized[j][preds[j]]), fontsize=6)
        imshow(inputs.cpu().data[j])
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    return

def imshow(inp, title=None, save_path=None, show=False):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.pause(0.1) # pause a bit so that plots are updated
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close('all')


if __name__ == '__main__':
    
    # Initialisation
    parser = ArgumentParser()
    parser.add_argument(
        'database',
        type=str,
        help='Video folder with category folders (no splitted), splitted folder is inferred by adding _split. ')
    args = parser.parse_args()

    session_path = os.path.join('plot_database_output', args.database)
    os.makedirs(session_path)

    numb_examples = 12
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.database, x)) for x in ['train', 'validation', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=numb_examples, shuffle=True) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    
    print_and_log('Classes: %s' % (', '.join(class_names)), log=log)

    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=', '.join([class_names[x] for x in classes[:numb_examples]]), save_path=os.path.join(session_path, 'samples_train.svg'))

    inputs, classes = next(iter(dataloaders['validation']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=', '.join([class_names[x] for x in classes[:numb_examples]]), save_path=os.path.join(session_path, 'samples_validation.svg'))

    inputs, classes = next(iter(dataloaders['test']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=', '.join([class_names[x] for x in classes[:numb_examples]]), save_path=os.path.join(session_path, 'samples_test.svg'))

    