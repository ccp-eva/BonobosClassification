import mahotas
import cv2
print('OpenCV version: ', cv2.__version__)
cv2.setNumThreads(0)
import os
import logging
import numpy as np
import sys
import platform
import matplotlib
# To be able to save figure using screen with matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

'''
Does what it says!
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)      
    return allFiles

'''
Does what it says!
'''
def count_frames_from_all_videos_in_folder(folder_path):
    N_frames=0
    for video_path in getListOfFiles(folder_path):
        cap = cv2.VideoCapture(video_path)
        N_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return N_frames

'''
Print and log functions
'''
def print_and_log(message, log=None):
    print(message)
    if log is not None:
        log.info(message)

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    return l

def close_log(log):
    if log is not None:
        x = list(log.handlers)
        for i in x:
            log.removeHandler(i)
            i.flush()
            i.close()

'''
Function to plot a progression bar in the terminal
'''
def progress_bar(count, total, title, completed=0, log=None):
    terminal_size = get_terminal_size()
    percentage = int(100.0 * count / total)
    length_bar = min([max([3, terminal_size[0] - len(title) - len(str(total)) - len(str(count)) - len(str(percentage)) - 10]),20])
    filled_len = int(length_bar * count / total)
    bar = '█' * filled_len + ' ' * (length_bar - filled_len)
    sys.stdout.write('%s [%s] %s %% (%d/%d)\r' % (title, bar, percentage, count, total))
    sys.stdout.flush()
    if completed:
        sys.stdout.write("\n")
        if log is not None:
            log.info('%s [%s] %s %% (%d/%d)' % (title, bar, percentage, count, total))


def get_terminal_size():
    '''
    This function determines the terminal size for different platforms
    '''
    def _get_terminal_size_windows():
        try:
            from ctypes import windll, create_string_buffer
            h = windll.kernel32.GetStdHandle(-12)
            csbi = create_string_buffer(22)
            res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
            if res:
                (bufx, bufy, curx, cury, wattr,
                left, top, right, bottom,
                maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
                sizex = right - left + 1
                sizey = bottom - top + 1
                return sizex, sizey
        except:
            pass

    def _get_terminal_size_tput():
        try:
            cols = int(subprocess.check_call(shlex.split('tput cols')))
            rows = int(subprocess.check_call(shlex.split('tput lines')))
            return (cols, rows)
        except:
            pass

    def _get_terminal_size_linux():
        def ioctl_GWINSZ(fd):
            try:
                import fcntl, termios, struct
                cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
                return cr
            except:
                pass
        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except:
                pass
        if not cr:
            try:
                cr = (os.environ['LINES'], os.environ['COLUMNS'])
            except:
                return None
        return int(cr[1]), int(cr[0])

    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os in ['Linux', 'Darwin'] or current_os.beginswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)      # default value
    return tuple_xy

'''
Some feature descriptors
'''
# Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# Color Histogram
def fd_histogram(image, mask=None, bins=8):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

'''
Some plot functions
'''
def plot_confusion_matrix(cm, classes, save_path, cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    '''
    
    acc = np.mean(np.array([cm[i,i] for i in range(len(cm))]).sum()/cm.sum()) * 100
    cm = cm / [max(tmp,1) for tmp in cm.sum(axis=1)]
    acc_2 = np.array([cm[i,i] for i in range(len(cm))])

    title = 'Accuracy of %.1f%%\n$\\mu$ = %.1f with $\\sigma$ = %.1f' % (acc, np.mean(acc_2)*100, np.std(acc_2)*100)
    if len(classes)>=12:
        plt.subplots(figsize=(12,12))
    elif len(classes)>=6:
        plt.subplots(figsize=(8,8))
    else:
        plt.subplots(figsize=(5,5))

    plt.imshow(cm.astype('float'), interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

'''
Plot train val curves
'''
def plot_curves(train_losses, train_acc, val_losses, val_acc, save_path='my_curves.png', plot_interval_point=1, plot_interval_line=1):
    '''
    This function plots the train and val curves with the given intervals.
    '''
    # Info for title
    N_epochs = len(train_losses)
    max_val_acc = max(val_acc)
    max_val_acc_idx = np.argmax(val_acc)
    max_train_acc = train_acc[max_val_acc_idx]

    min_val_loss = min(val_losses)
    min_val_loss_idx = np.argmin(val_losses)
    min_train_loss = train_losses[min_val_loss_idx]

    font = {'family' : 'cmr10', 'size'   : 13}
    axes = {'formatter.use_mathtext': True}
    plt.rc('font', **font)
    plt.rc('axes', **axes)

    host = host_subplot(111, axes_class=AA.Axes)
    host.clear()
    par = host.twinx()

    par.axis["right"].toggle(all=True)

    host.set_xlim(0, N_epochs)
    host.set_ylim(0, max(max(train_losses), max(val_losses)))
    par.set_ylim(0, 1)
    x_ticks_points = np.arange(N_epochs, step=plot_interval_point)
    x_ticks_lines = np.arange(N_epochs, step=plot_interval_line)
    host.set_ylim(np.min([np.min(train_losses)-0.01, np.min(val_losses)-0.01]), np.max([np.max(train_losses)+0.01, np.max(val_losses)+0.01]))

    host.set_title("Max val acc %.2f%% with train acc %.2f%% at epoch %d\nMin val loss %.2f with train loss %.2f at epoch %d" % (max_val_acc*100, max_train_acc*100, max_val_acc_idx, min_val_loss, min_train_loss, min_val_loss_idx))
    host.set_xlabel("Epochs")
    host.set_ylabel("Loss")
    par.set_ylabel("Accuracy")

    l1 = host.plot(x_ticks_points, train_losses[0::plot_interval_point], '^', color='tomato', label="RGB Train loss", alpha=1, markersize=8)
    p1 = host.plot(x_ticks_lines, train_losses[0::plot_interval_line], color='tomato', alpha=0.5)
    l2 = host.plot(x_ticks_points, val_losses[0::plot_interval_point], 'gv', label="RGB Val loss", alpha=1, markersize=8)
    p2 = host.plot(x_ticks_lines, val_losses[0::plot_interval_line], color='g', alpha=0.5)
    l3 = par.plot(x_ticks_points, val_acc[0::plot_interval_point], 'b>',label="RGB Val accuracy", alpha=1, markersize=8)
    p3 = par.plot(x_ticks_lines, val_acc[0::plot_interval_line], color='b', alpha=0.5)
    l4 = par.plot(x_ticks_points, train_acc[0::plot_interval_point], '<', color='crimson', label="RGB Train accuracy", alpha=1, markersize=8)
    p4 = par.plot(x_ticks_lines, train_acc[0::plot_interval_line], color='crimson', alpha=0.5)

    host.legend(loc='center right', ncol=1, fancybox=False, shadow=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

'''
Stats on the datasets splitted (need to fit naming convention)
'''
def plot_data_distribution():
    print("Stats splited dataset")
    fig = plt.figure()
    font = {'family' : 'cmr10', 'size'   : 13}
    axes = {'formatter.use_mathtext': True}
    plt.rc('font', **font)
    plt.rc('axes', **axes)
    ax = fig.add_axes([0,0,1,1])
    alpha = 0.6
    list_categories = os.listdir('ROI_S0')
    check_pass = False
    for database in ['ROI_S0_split', 'ROI_S0.5_split']:
        total_samples_per_set = {'train': 0, 'validation': 0, 'test': 0}
        print('In %s' % (database))
        # For plotting
        dict_dataset = {}
        for dataset in total_samples_per_set.keys():
            dict_dataset[dataset] = dict.fromkeys(list_categories, 0)

        for category in list_categories:
            print("For %s" % (category))
            for dataset in total_samples_per_set.keys():
                nb_samples = 0
                for video in os.listdir(os.path.join(database, dataset, category)):
                    nb_samples += len(os.listdir(os.path.join(database, dataset, category, video)))
                total_samples_per_set[dataset] += nb_samples
                dict_dataset[dataset][category] += nb_samples
                print("\t%s: %d samples" % (dataset, nb_samples))
        print("Total number of samples per set: %s\n" % (str(total_samples_per_set)))

        if not check_pass:
            check_pass = True
            samples_per_id = [dict_dataset['train'][key]+dict_dataset['validation'][key]+dict_dataset['test'][key] for key in list_categories]
            sorted_list_of_category = [x for _,x in sorted(zip(samples_per_id,list_categories))]

        X = np.arange(len(list_categories))
        
        ax.bar(X + 0.00, [dict_dataset['train'][key] for key in sorted_list_of_category], width = 0.25, alpha=alpha)
        ax.bar(X + 0.25, [dict_dataset['validation'][key] for key in sorted_list_of_category], width = 0.25, alpha=alpha)
        ax.bar(X + 0.50, [dict_dataset['test'][key] for key in sorted_list_of_category], width = 0.25, alpha=alpha)
        alpha+=0.3
        plt.gca().set_prop_cycle(None) # reset color cycle
    
    ax.set_ylabel('Samples')
    ax.set_title('Samples repartition')
    ax.set_xticks(X+0.25, sorted_list_of_category)
    ax.legend(labels=total_samples_per_set.keys())
    plt.savefig('samples_distribution.svg',bbox_inches='tight')