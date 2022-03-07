import numpy as np
import os
import cv2
print('OpenCV version: ', cv2.__version__)
from argparse import ArgumentParser
import json
import shutil

def extract_roi_from_video(video_file, detection_file, output_images, score_thr, no_roi):
    if not os.path.exists(output_images):
        os.makedirs(output_images)
    cap = cv2.VideoCapture(video_file)

    with open(detection_file) as f:
        detection_data = json.load(f)
        # for frame_id, frame_data in detection_data.items():
        score_last = 0
        for instance_detected in detection_data:
            image_id = instance_detected['image_id']
            cap.set(cv2.CAP_PROP_POS_FRAMES, image_id)
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame %d" % image_id)
                continue
            bbox = instance_detected['bbox']
            score = instance_detected['score']
            # If two instances are detected in the same frame, we take the one with the highest score.
            # Not best method when not using ROI
            # Known issue: the rotation of the video in metadata is not taken into account - lead to error
            if (not os.path.exists(os.path.join(output_images, '%06d.png' % image_id)) or score > score_last) and score >= score_thr:
                if no_roi:
                    cv2.imwrite(os.path.join(output_images, '%06d.png' % image_id), frame)
                else:
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    roi = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(output_images, '%06d.png' % image_id), roi)
                score_last = score            


def extract_roi_from_folder_video(category_path, detection_path, output_path, score_thr=0, no_roi=False):
    list_videos = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path,f))]
    print("There are %d videos:" % len(list_videos), list_videos)
    for video in list_videos:
        print('Processing video %s' % video)
        video_file = os.path.join(category_path, video)
        detection_file = os.path.join(detection_path, os.path.splitext(video)[0] + ".json")
        # ## For creation of the dataset from Github preparation
        # detection_file = os.path.join(detection_path, os.path.splitext(video)[0] + "_detections.json")
        # category = os.path.basename(category_path)
        # os.makedirs(os.path.join('database/videos/', category), exist_ok=True)
        # os.makedirs(os.path.join('database/detections/', category), exist_ok=True)
        # shutil.copyfile(video_file, os.path.join('database/videos/', category, video))
        # shutil.copyfile(detection_file, os.path.join('database/detections/', category, os.path.splitext(video)[0] + ".json"))
        # ##
        output_images = os.path.join(output_path, os.path.splitext(video)[0])
        os.makedirs(output_images, exist_ok=True)
        extract_roi_from_video(video_file, detection_file, output_images, score_thr, no_roi)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--video_input',
        type=str,
        default='database/videos/',
        help='Video folder with category folders. ')
    parser.add_argument(
        '--detection_input',
        type=str,
        default='database/detections/',
        help='Json files with the detections.')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='ROI_S0',
        help='Output where the segmentation will be saved following same tree than the video folder. ')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0,
        help='Score of the detection to consider. Per default, all detections are considered. ')
    parser.add_argument(
        '--no-roi',
        action='store_true',
        help='Do not use roi from detection. ')
    
    print('Initialisation')
    args = parser.parse_args()
    list_categories = [f for f in os.listdir(args.video_input) if os.path.isdir(os.path.join(args.video_input,f))]
    # print("There are %d categories:" % len(list_categories), list_categories)

    os.makedirs(args.output_folder, exist_ok=True)
    for category in list_categories:
        print('Processing category %s' % category)
        category_path = os.path.join(args.video_input, category)
        detection_path = os.path.join(args.detection_input, category)
        output_path = os.path.join(args.output_folder, category)
        os.makedirs(output_path, exist_ok=True)
        extract_roi_from_folder_video(category_path, detection_path, output_path, score_thr=args.score_thr, no_roi=args.no_roi)

    print("Stats on your freasly created dataset")
    total_samples = 0
    for category in list_categories:
        nb_samples = 0
        for video in os.listdir(os.path.join(args.output_folder, category)):
            nb_samples += len(os.listdir(os.path.join(args.output_folder, category, video)))
        print("For %s there are %d samples" % (category, nb_samples))
        total_samples+=nb_samples

if __name__ == '__main__':
    main()
