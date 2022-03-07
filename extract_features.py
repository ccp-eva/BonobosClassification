#-----------------------------------
# GLOBAL FEATURE EXTRACTION - see https://gogul.dev/software/image-classification-python
#-----------------------------------
from utils import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import h5py
from argparse import ArgumentParser
import random
random.seed(0)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        'database_path',
        type=str,
        help='Input where the segmentation will be saved following same tree than the video folder. ')
    parser.add_argument(
        'database_features',
        type=str,
        help='Output where the features will be saved. ')
    # parser.add_argument(
    #     '--fixed-size',
    #     type=str,
    #     help='Output where the features will be saved. ')

    print('Initialisation')
    args = parser.parse_args()
    os.makedirs(args.database_features, exist_ok=True)
    #--------------------
    # tunable-parameters
    #--------------------
    images_per_class = 1500
    # fixed_size       = tuple((250, 250))
    h5_data          = os.path.join(args.database_features,'data.h5')
    h5_labels        = os.path.join(args.database_features,'labels.h5')
    bins             = 8

    # get the training labels
    train_labels = os.listdir(args.database_path)

    # sort the training labels
    train_labels.sort()
    print(train_labels)

    # empty lists to hold feature vectors and labels
    global_features = []
    labels          = []

    # loop over the training data sub-folders
    for category in train_labels:
        # join the training data path and each species training folder
        list_of_samples = getListOfFiles(os.path.join(args.database_path, category))
        random.shuffle(list_of_samples)

        # loop over the images in each sub-folder
        # for frame_path in list_of_samples[:images_per_class]:
        for frame_path in list_of_samples:
            # read the image and resize it to a fixed-size
            image = cv2.imread(frame_path)
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

            # update the list of labels and feature vectors
            labels.append(category)
            global_features.append(global_feature)

        print("[STATUS] processed folder: {}".format(category))

    print("[STATUS] completed Global Feature Extraction...")

    # get the overall feature vector size
    print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

    # get the overall training label size
    print("[STATUS] training Labels {}".format(np.array(labels).shape))

    # encode the target labels
    targetNames = np.unique(labels)
    le          = LabelEncoder()
    target      = le.fit_transform(labels)
    print("[STATUS] training labels encoded...")

    # scale features in the range (0-1)
    scaler            = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print("[STATUS] feature vector normalized...")

    print("[STATUS] target labels: {}".format(target))
    print("[STATUS] target labels shape: {}".format(target.shape))

    # save the feature vector using HDF5
    h5f_data = h5py.File(h5_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File(h5_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    print("[STATUS] end of feature extraction..")


if __name__ == '__main__':
    main()