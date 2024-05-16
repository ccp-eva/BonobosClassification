#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------
from utils import *
import h5py
import glob
import warnings
import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser

def import_features_label(path_features, path_labels):
    # import the feature vector and labels from the whole dataet
    h5f_features  = h5py.File(path_features, 'r')
    h5f_label = h5py.File(path_labels, 'r')
    features_string = h5f_features['dataset_1']
    labels_string   = h5f_label['dataset_1']
    features = np.array(features_string)
    labels   = np.array(labels_string)
    h5f_features.close()
    h5f_label.close()
    return features, labels

def plot_cm_boxplot(session_path, models, y_true, y_predicted, list_of_labels, dataset_name):
    ''' Plot confusion matrix and boxplot of the predictions for each model '''
    results = []
    names = []
    # For confusion matrix
    for name, model in models:
        cm = confusion_matrix(y_true, y_predicted[name], labels=list_of_labels)
        plot_confusion_matrix(cm, list_of_labels, os.path.join(session_path, dataset_name, 'cm_'+name+'.png'))
        names.append(name)
        results.append([np.mean(np.array([cm[i,i] for i in range(len(cm))]).sum()/cm.sum()) * 100])

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Machine Learning algorithm comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig(os.path.join(session_path, 'ml_algo_comparison_%s.png' % (dataset_name)))
    plt.close('all')
    return 1

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = ArgumentParser()
    parser.add_argument(
        'database',
        type=str,
        help='Video folder with category folders (no splitted), splitted folder is inferred by adding _split. ')

    print('Initialisation')
    args = parser.parse_args()

    #--------------------
    # tunable-parameters
    #--------------------
    num_trees = 100
    seed      = 9
    fixed_size       = tuple((250, 250))

    # Split sets
    train_path  = os.path.join('%s_split' % (args.database), 'train')
    validation_path    = os.path.join('%s_split' % (args.database), 'validation')
    test_path   = os.path.join('%s_split' % (args.database), 'test')

    # Features and labels
    h5_data    = os.path.join('features_%s' % (args.database), 'data.h5')
    h5_labels  = os.path.join('features_%s' % (args.database), 'labels.h5')
    h5_data_train    = os.path.join('features_%s' % (args.database), 'train', 'data.h5')
    h5_labels_train  = os.path.join('features_%s' % (args.database), 'train', 'labels.h5')
    scoring    = "accuracy"
    session_path = os.path.join('ml_classification_output', args.database, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(session_path)
    log = setup_logger('my_log', os.path.join(session_path, 'log.txt'))
    os.makedirs(os.path.join(session_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(session_path, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(session_path, 'test'), exist_ok=True)

    # get the training labels
    list_of_labels = os.listdir(train_path)
    # sort the training labels
    list_of_labels.sort()

    # create all the machine learning models
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))

    # variables to hold the results and names
    results = []
    names   = []

    # import the feature vector and labels
    # Note: this is not efficient because we have to have the data twice
    global_features, global_labels = import_features_label(h5_data, h5_labels)
    train_features, train_labels = import_features_label(h5_data_train, h5_labels_train)

    # verify the shape of the feature vector and labels
    print_and_log("[STATUS] features shape: {}".format(global_features.shape), log)
    print_and_log("[STATUS] labels shape: {}".format(global_labels.shape), log)
    print_and_log("[STATUS] splitted train and test data...", log)
    print_and_log("Train data  : {}".format(global_features.shape), log)
    print_and_log("Train labels: {}".format(global_labels.shape), log)

    # 10-fold cross validation on the whole dataset
    print_and_log("[STATUS] 10-fold cross validation started...", log)
    for name, model in models:
        # kfold = KFold(n_splits=10, random_state=seed)
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, global_features, global_labels, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print_and_log(msg, log)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Machine Learning algorithm comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig(os.path.join(session_path,'ml_algo_comparison_cross_validation.png'))
    plt.close('all')

    np.save(os.path.join(session_path, 'models_global'), models, allow_pickle=True)

    # -----------------------------------
    # TESTING OUR MODELS on validation and test data
    # -----------------------------------
    # models = np.load('models.npy', allow_pickle=True)
    # print_and_log('[STATUS] Load OK', log)

    # create all the machine learning models
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))

    print_and_log('[STATUS] Train models and prediction on the train set', log)
    y_true = []
    for label_idx in train_labels:
        y_true.append(list_of_labels[label_idx])
    y_predicted = {}
    for name, model in models:
        print_and_log('[STATUS] %s model' % (name), log)
        model.fit(train_features, train_labels)
        predictions = model.predict(train_features)
        y_predicted[name] = []
        for label_idx in predictions:
            y_predicted[name].append(list_of_labels[label_idx])

    np.save(os.path.join(session_path, 'models_trainsplit'), models, allow_pickle=True)
    plot_cm_boxplot(session_path, models, y_true, y_predicted, list_of_labels, 'train')
    
    for dataset_path in [validation_path, test_path]:
        # For confusion matrix
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        y_true = []
        y_predicted = {}
        print_and_log("[STATUS] Inference on %s" % (dataset_name), log)
        for name, model in models:
            y_predicted[name] = []
            
        # loop through the test images
        for label in list_of_labels:
            print_and_log("\t[STATUS] Evaluation on %s" % (label), log)
            image_paths = getListOfFiles(os.path.join(dataset_path, label))
            
            for image_path in image_paths:
                # read the image
                image = cv2.imread(image_path)

                # # resize the image
                # image = cv2.resize(image, fixed_size)

                ####################################
                # Global Feature extraction
                ####################################
                fv_hu_moments = fd_hu_moments(image)
                fv_haralick   = fd_haralick(image)
                fv_histogram  = fd_histogram(image)

                ###################################
                # Concatenate global features
                ###################################
                global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
                y_true.append(label)

                # scale features in the range (0-1)
                # scaler = MinMaxScaler(feature_range=(0, 1))
                # rescaled_feature = scaler.fit_transform(global_feature.reshape(1,-1))
                rescaled_feature = global_feature

                for name, model in models:
                    # predict label of test image
                    prediction = model.predict(rescaled_feature.reshape(1,-1))[0]
                    y_predicted[name].append(list_of_labels[prediction])

                    # # show predicted label on image
                    # cv2.putText(image, list_of_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

                    # # display the output image
                    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # plt.show()
        print_and_log("[STATUS] Confusion Matrix creation.", log)
        plot_cm_boxplot(session_path, models, y_true, y_predicted, list_of_labels, dataset_name)

    print_and_log('[STATUS] End of the program', log)
    close_log(log)
