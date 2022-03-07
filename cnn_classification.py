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


def reset_session(seed=0):
    ''' Reset Pytorch session'''
    gc.collect()
    torch.manual_seed(seed)

def imshow(inp, title=None, save_path=None, show=False):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
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
        

def visualize_model(model, save_path, num_images=9, show=False):
    # was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    fig.suptitle('Model Predictions: Ground Truth-Prediction (score/normalized)', fontsize=8)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['validation']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            output_normalized = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//3, 3, images_so_far)
                ax.axis('off')                
                ax.set_title('%s-%s (%.2f/%.2f)' % (class_names[labels[j]], class_names[preds[j]], outputs[j][preds[j]], output_normalized[j][preds[j]]), fontsize=6)
                imshow(inputs.cpu().data[j], show=show)
                if images_so_far == num_images:
                    # model.train(mode=was_training)
                    # set the spacing between subplots
                    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.3)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close('all')
                    return
        # model.train(mode=was_training)

def test_model(model, criterion, log=None, model_path='model'):
    since = time.time()
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    idx = 0
    phase = 'test'
    y_true = []
    y_predicted = []

    # Iterate over data.
    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            idx+=1
            progress_bar(idx, len(dataloaders[phase]), 'Test phase')
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward - do not track history
            # with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # confusion matrix
            y_true.extend([dataloaders[phase].dataset.classes[i] for i in labels.data.cpu().numpy()])
            y_predicted.extend([dataloaders[phase].dataset.classes[i] for i in preds.cpu().numpy()])

    test_loss = running_loss / dataset_sizes[phase]
    test_acc = running_corrects.double() / dataset_sizes[phase]
    time_elapsed = time.time() - since
    cm = confusion_matrix(y_true, y_predicted, labels=dataloaders[phase].dataset.classes)
    plot_confusion_matrix(cm, dataloaders[phase].dataset.classes, '%s_cm_%s_.svg' % (model_path, phase))

    progress_bar(
        idx,
        len(dataloaders[phase]),
        'Test done in %.0fm %.0fs with Loss: %.4f Acc: %.4f' % (time_elapsed // 60, time_elapsed % 60, test_loss, test_acc),
        completed=1,
        log=log)

    return model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, log=None, model_path='model'):
    since = time.time()
    best_model_wts_loss = copy.deepcopy(model.state_dict())
    best_model_wts_acc = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0

    # For plot
    history = {'train_loss': [], 'train_acc': [], 'validation_loss': [], 'validation_acc': []}

    for epoch in range(num_epochs):
        print_and_log('Epoch %d/%d' % (epoch, num_epochs - 1), log=log)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0
            running_corrects = 0
            idx = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                idx+=1
                progress_bar(idx, len(dataloaders[phase]), 'Phase %s' % (phase))
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            history['%s_loss' % (phase)].append(epoch_loss)
            history['%s_acc' % (phase)].append(epoch_acc)

            progress_bar(idx, len(dataloaders[phase]), 'Phase %s done with Loss: %.4f Acc: %.4f' % (phase, epoch_loss, epoch_acc), completed=1, log=log)

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts_acc = copy.deepcopy(model.state_dict())
                print_and_log('Model saved with regard to acc', log=log)

            # deep copy the model
            if phase == 'validation' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts_loss = copy.deepcopy(model.state_dict())
                print_and_log('Model saved with regards to loss', log=log)

    time_elapsed = time.time() - since
    print_and_log('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), log=log)
    print_and_log('Best validation Acc: {:4f}'.format(best_acc), log=log)

    # plot history
    plot_curves(history['train_loss'], history['train_acc'], history['validation_loss'], history['validation_acc'], save_path='%s_curves.svg' % (model_path))

    # load and locally save best model weights
    model_loss = copy.deepcopy(model)
    model.load_state_dict(best_model_wts_acc)
    model_loss.load_state_dict(best_model_wts_loss)
    torch.save(best_model_wts_acc, '%s_acc_weigths.pth' % (model_path))
    torch.save(best_model_wts_loss, '%s_loss_weigths.pth' % (model_path))

    return model, model_loss

if __name__ == '__main__':
    
    # Initialisation
    parser = ArgumentParser()
    parser.add_argument(
        'database',
        type=str,
        help='Video folder with category folders (no splitted), splitted folder is inferred by adding _split. ')
    parser.add_argument(
        '--weighted-loss',
        action='store_true',
        help='Use weighted loss according to data distribution. ')
    parser.add_argument(
        '--model-wts-path',
        type=str,
        default='./checkpoints/resnet18-5c106cde.pth',
        help='Pretrained weights. ')
    args = parser.parse_args()

    reset_session()
    session_path = os.path.join('cnn_classification_output', args.database, 'weighted_loss_%s' % (args.weighted_loss), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(session_path)
    log = setup_logger('logger_name', os.path.join(session_path, 'log.txt'))
    num_epochs = 100
    step_size = 20
    lr = 0.001
    lr = 0.00001 # Descreased by 100 for weighted loss
    batch_size=64
    plot_datasets = False
    print_and_log('With the parameters: lr=%g, num_epochs=%d, step_size=%d' % (lr, num_epochs, step_size), log=log)

    model_wts = torch.load(args.model_wts_path)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize([224,224]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.database, x),
                                            data_transforms[x])
                    for x in ['train', 'validation', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=10)
                for x in ['train', 'validation', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    
    np.save(os.path.join(session_path, 'class_names.npy'), class_names)

    print_and_log('Classes: %s' % (', '.join(class_names)), log=log)

    # CPU or GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    if args.weighted_loss: # Would be equal to one if even distribution accross the classes
        class_weights = torch.from_numpy(len(image_datasets['train'])/(np.array([len(getListOfFiles(os.path.join(args.database,'train',name))) for name in class_names])+1)/len(class_names)).float().to(device)
    else:
        class_weights = None

    # Get a batch of training data and plot the first 10 images
    if plot_datasets:
        numb_examples = 8
        inputs, classes = next(iter(dataloaders['train']))
        out = torchvision.utils.make_grid(inputs[:numb_examples])
        imshow(out, title=', '.join([class_names[x] for x in classes[:numb_examples]]), save_path=os.path.join(session_path, 'samples_train.svg'))

        inputs, classes = next(iter(dataloaders['validation']))
        out = torchvision.utils.make_grid(inputs[:8])
        imshow(out, title=', '.join([class_names[x] for x in classes[:numb_examples]]), save_path=os.path.join(session_path, 'samples_validation.svg'))

        inputs, classes = next(iter(dataloaders['test']))
        out = torchvision.utils.make_grid(inputs[:8])
        imshow(out, title=', '.join([class_names[x] for x in classes[:numb_examples]]), save_path=os.path.join(session_path, 'samples_test.svg'))

    # Pretrained model finetune on our database - all weights are updated
    print_and_log('ConvNet finetuned', log=log)
    model_ft = models.resnet18()
    model_ft.load_state_dict(model_wts)

    num_ftrs = model_ft.fc.in_features

    # Set the last layer lenght to the number of classes
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)

    ## Train and evaluate ##
    model_ft, model_ft_loss = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, log=log, model_path=os.path.join(session_path, 'model_ft'))

    # Test the model
    test_model(model_ft, criterion, log=log, model_path=os.path.join(session_path, 'model_ft'))
    visualize_model(model_ft, os.path.join(session_path, 'model_ft_vis.svg'))

    test_model(model_ft_loss, criterion, log=log, model_path=os.path.join(session_path, 'model_ft_loss'))
    visualize_model(model_ft_loss, os.path.join(session_path, 'model_ft_loss_vis.svg'))

    ########################################
    ## ConvNet as fixed feature extractor ##
    ########################################
    reset_session()
    print_and_log('\n\nConvNet as fixed feature extractor', log=log)
    model_conv = torchvision.models.resnet18()
    model_conv.load_state_dict(model_wts)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features

    # Set the last layer lenght to the number of classes
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=0.1)
    model_conv, model_conv_loss = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs, log=log, model_path=os.path.join(session_path, 'model_fextractor'))

    # Test the model
    test_model(model_conv, criterion, log=log, model_path=os.path.join(session_path, 'model_fextractor'))
    visualize_model(model_conv, os.path.join(session_path, 'model_fextractor_vis.svg'))

    test_model(model_conv_loss, criterion, log=log, model_path=os.path.join(session_path, 'model_fextractor_loss'))
    visualize_model(model_conv_loss, os.path.join(session_path, 'model_fextractor_loss_vis.svg'))
    
    print_and_log(message='\n\nFinished', log=log)
    close_log(log)
    