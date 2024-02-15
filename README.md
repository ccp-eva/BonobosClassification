# Introduction
The Bonobos classification repository aims to reproduce the work of Bonobos individual classification from an auto-generated dataset using pre-trained models and a light annotation procedure.
The aim is to have a reproducible pipeline in order to build a primate identification tool in zoos or sanctuaries for various applications.

# Dataset
## Presentation
The provided dataset was recorded at the Zoo Berlin using a digital camcorder Panasonic HC-V757 and a cheap Logitech webcam, both of resolution 1280x720 at 30 fps. The videos can be assimilated to focal observation consisting of observing one particular individual and observing his/her actions and interactions. This may lead to having several individuals in the field of the camera, or none because of obstruction, camera manipulation, or the case of not having the individual in the webcam's field of view. No spatial information was annotated, nor was the presence of the individual in the field of the camera if several individuals were in the field of view. In this particular enclosure there are seven individuals  of different gender and age (gender/year of birth): Matayo (male/2019), Monyama (female/2010), Opala (female/1998), Santi (male/1981), Limbuko (male/1995), Leki (female/2014) and Samani female/2020). Samani was not incorporated into the dataset because of her constant proximity to her mother Monyama. A total of 100 videos inequitably distributed across six bonobo individuals is considered. The hand annotation consists of telling on which bonobo the recording is focusing on. The automatic annotation is based on OpenMMLab macaque detector.

![](samples_database.png)

## Download
The dataset is available on our Nextcloud instance.
You may download it using your terminal and check its consistency.
A step-by-step instruction.

1. Make sure to have cloned this repo:
```
git clone https://github.com/ccp-eva/BonobosClassification.git
cd BonobosClassification
```

2. Download the database.zip file from our Nextcloud:
```
curl -X GET -u "MBby5AstWe9JiEY:TxsxnWcoEa" -H 'X-Requested-With: XMLHttpRequest' 'https://share.eva.mpg.de/public.php/webdav/database.zip' -o database.zip
```
Alternatively, you can use your browser using this [link](https://share.eva.mpg.de/index.php/s/MBby5AstWe9JiEY) and this password: TxsxnWcoEa .

3. Check its content with md5sum:
```
md5sum -c database.md5
```

4. unzip the file to the database folder:
```
unzip -d database database.zip
```

You should obtain a database folder with subfolders videos and detections and subsubfolder with the bonobos' name.

## Datasets generation

Different databases may be generated according to the source files. The script `create_database.py` is meant to create different databases according to the ROI consideration and score threshold. `python3 create_database.py` will create a ROI_S0 database which takes into account the ROI and all frames with bonobos detected regardless of the dtectin score. `python3 create_database.py -h`  for more options.
```
python3 create_database.py --video_input 'database/videos/ --detection_input database/detections --output_folder ROI_S0 --score-thr 0
```

Finally, the dataset may be split into the train, validation and test sets using `split_database.py`. The script will create another folder with the name of the database provided + "_split" with the different sets. You may check the distribution used in our work in [data_distribution.txt](data_distribution.txt).
```
python3 split_database.py ROI_S0
```

# Bonobo Individual Classification
## ML classification

To run different ML algorithms (not deep learned), you may first extract the features of each dataset by running `extract_feature.py` on the non split dataset (e.g. `ROI_S0`) and the train split (e.g. `ROI_S0_split/train`).
Then the `ml_classification.py` script will automatically run the classification on the non-split and split dataset.
```
python3 extract_feature.py ROI_S0
python3 extract_feature.py ROI_S0_split/train
python3 ml_classification.py ROI_S0
```

## ResNet classification.

ResNet classification may be run by calling the script `cnn_classification.py` on the generated split datasets. More options by calling `python3 cnn_classification.py -h`. The model will use the pre-trained weights of resnet18 provided by Pytorch. Make sure to download the weights before continuing: 
```
wget -P checkpoints https://download.pytorch.org/models/resnet18-5c106cde.pth
python3 cnn_classification.py ROI_S0_split
```

# To cite this work

In pipeline. Get in touch with us directly if you have any questions: pierre_etienne_martin (at) eva.mpg.de.
