import customdataset
import os
from facenet_pytorch import MTCNN, extract_face
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import utils, transforms
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
import pandas as pd
import enums
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import torch


class FaceExtractor:
  def __init__(self, config, tqdm=None):
    self.config = config
    self.tqdm = tqdm

  def extract_faces(self, output_base, summary_file_name, annotation_file, limit=None):
      # mtcnn = MTCNN(keep_all=False, selection_method='center_weighted_size', select_largest=True)
      mtcnn = MTCNN(keep_all=False, select_largest=True)
      transform = transforms.Compose([transforms.ToTensor(), T.ToPILImage()])
      dataset = customdataset.OriginalDataset(self.config, transform=transform)
      dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True, collate_fn=dataset.get_image_from)

      summary_cols = ['filename', 'class', 'num_faces', 'boxes', 'points', 'probs']
      summary = []
      annotation = []
      files_to_process = limit if limit is not None else len(dataloader)
      pbar = None if self.tqdm is None else self.tqdm(total=files_to_process, position=0, leave=True, unit='files')
      for batch_idx, samples in enumerate(dataloader):
          # if batch_idx % 1000 == 0:
          #     print(f'Processed {batch_idx} files...')
          if limit is not None and batch_idx > limit:
                  break
          img, label, filename = samples
          if pbar is not None:
              pbar.set_description('Processing {:<14}'.format(filename))
              pbar.update(1)

          filename = os.path.basename(filename)
          file_primary_name, _ = os.path.splitext(filename)

          # Get bounding boxes as well as facial keypoints from model
          boxes, probs, points = mtcnn.detect(img, landmarks=True)

          # Annotate image with bounding boxes + keypoints
          img_draw = img.copy()
          draw = ImageDraw.Draw(img_draw)
          Path(f'{output_base}/c{label}').mkdir(parents=True, exist_ok=True)
          if points is not None and boxes is not None:
              for i, (box, point) in enumerate(zip(boxes, points)):
                  draw.rectangle(box.tolist(), width=1)
                  for p in point:
                      draw.ellipse((p - 2).tolist() + (p + 2).tolist(), fill='red')
                  # Save the extracted face as another file
                  extract_face(img, box, save_path=f'{output_base}/c{label}/{file_primary_name}_{i}.png')

          # save a copy of the annotated image.
          img_draw.save(f'{output_base}/c{label}/{file_primary_name}_annotated.png')

          # Save summary of the file.
          summary.append([filename, label, len(boxes) if boxes is not None else 0, boxes, points, probs])
          annotation.append([filename, label, len(boxes) if boxes is not None else 0, enums.SampleType.IGNORED])

      # Summarize the run.
      df_summary = pd.DataFrame(summary, columns=summary_cols)
      df_summary.to_csv(f'{output_base}/{summary_file_name}', index=False)

      df_annotation = pd.DataFrame(annotation, columns=self.config.ANNOTATION_FILE_COLS)
      df_annotation.to_csv(annotation_file, index=False)
      if pbar is not None:
          pbar.close()
      return df_summary

class SampleSplitter:
    def __init__(self, config, face_config, pose_config, tqdm):
        self.face_config = face_config
        self.pose_config = pose_config
        self.config = config
        self.tqdm = tqdm

    def sample(self, classes, samples_per_class, out_file):
        # Load the face annotation file
        pd_orig_data = pd.read_csv(self.config.ANNOTATION_FILE) # Cols: filename, class, num_faces
        # pd_orig_data['sample_type'] = [enums.SampleType.IGNORED]*pd_orig_data.shape[0]
        print(f'Total samples: {pd_orig_data.shape[0]}')
        samples = {}

        [total_num, training_num, test_num, validation_num] = samples_per_class
        num_classes = len(classes)

        # Sample each class
        pbar = None if self.tqdm is None else self.tqdm(total=total_num * num_classes, position=0, leave=True, unit='file')
        for i, label in enumerate(classes):
            # self, config, face_config, pose_config, 
            #      sample_type=enums.SampleType.ALL, image_types=None,
            #      transformers=None, label=None, should_load_images=True
            dataset = customdataset.MainDataset(self.config, self.face_config, self.pose_config,
                                                sample_type=enums.SampleType.WITH_JUST_ONE_FACE,
                                                label=label, should_load_images=False)
            # Random shuffling is important
            dataloader = DataLoader(dataset, num_workers=0, batch_size=1,
                                    shuffle=True, collate_fn=dataset.get_image_from)

            for sample_num, sample in enumerate(dataloader):
                _, _, filename = sample
                if pbar is not None:
                    pbar.set_description(f'Processing {filename}')
                    pbar.update(1)
                if filename in samples:
                    print(f'WARN: duplicate filename {filename}')
                if sample_num < training_num:
                    samples[filename] = [label, enums.SampleType.TRAIN]
                elif sample_num < training_num + test_num:
                    samples[filename] = [label, enums.SampleType.TEST]
                elif sample_num < training_num + test_num + validation_num:
                    samples[filename] = [label, enums.SampleType.VALIDATION]
                else:
                    break
        if pbar is not None:
            pbar.close()
        print('Validating and saving the split-up...')
        if not len(samples) == num_classes * total_num:
            print(f'Did not get enough samples. Expecting {num_classes * total_num}; got {len(samples)}')
        for i, row in pd_orig_data.iterrows():
            filename = pd_orig_data.at[i, 'filename']
            if filename in samples:
                [_, cur_sample_type] = samples[filename]
            else:
              cur_sample_type = enums.SampleType.IGNORED
            pd_orig_data.at[i, 'sample_type'] = cur_sample_type
        print(f"Created {pd_orig_data[pd_orig_data['sample_type'] == 1].shape[0]} training samples")
        print(f"Created {pd_orig_data[pd_orig_data['sample_type'] == 2].shape[0]} validation samples")
        print(f"Created {pd_orig_data[pd_orig_data['sample_type'] == 3].shape[0]} testing samples")
        print(f"Left with {pd_orig_data[pd_orig_data['sample_type'] == 4].shape[0]} unused samples")
        pd_orig_data.to_csv(out_file, header=True, index=False)

class PoseExtractor:
    def __init__(self, config, face_config, pose_config, tqdm=None, model=None):
        self.config = config
        self.face_config = face_config
        self.pose_config = pose_config
        self.tqdm = tqdm
        if model is None:
            print('Loading model...')
            model = tf.saved_model.load(pose_config.SAVED_MODEL_FOLDER)
        self.infer = model.signatures["serving_default"]

    def extract_poses(self, output_base, summary_file_name, limit=None, labels = None):
        def get_pbar():
            full_dataset = customdataset.MainDataset(self.config, self.face_config, self.pose_config,
                                        should_load_images=False,
                                        sample_type=enums.SampleType.TRAIN_TEST_VALIDATION)
            universe = len(full_dataset)
            total_sample_count = universe if limit is None else min(universe, limit)
            pbar = None if self.tqdm is None else self.tqdm(total=total_sample_count, position=0, leave=True)
            return pbar

        if labels is None:
            labels = self.config.class_dict.keys()
        total_images = 0

        pbar = get_pbar()
        for label in labels:
            if limit is not None and total_images > limit:
                    break
            dataset = customdataset.MainDataset(self.config, self.face_config, self.pose_config,
                                                labels=[label], should_load_images=False,
                                                sample_type=enums.SampleType.TRAIN_TEST_VALIDATION)
            dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False, collate_fn=dataset.get_image_from)
            for batch_idx, samples in enumerate(dataloader):
                if batch_idx > 0 and batch_idx % 100 == 0:
                    if pbar is None:
                        print(f'Processed {label}:{batch_idx} files...')
                        print(f'Extracted from {total_images} files...')
                    else:
                        desc = f'Processing images labeled \'{self.config.class_dict[label]}\''
                        pbar.set_description(desc)
                if limit is not None:
                    if total_images > limit:
                        break
                _, label, orig_filename = samples
                filename = os.path.basename(orig_filename)
                file_primary_name, _ = os.path.splitext(os.path.basename(filename))
                op_pose = f'{output_base}/c{label}/{file_primary_name}_pose.png'
                op_annotated = f'{output_base}/c{label}/{file_primary_name}_annotated.png'
                op_cropped_pose = f'{output_base}/c{label}/{file_primary_name}_pose_cropped.png'
                files_exist = os.path.exists(op_pose) and os.path.exists(op_annotated) and os.path.exists(op_cropped_pose)
                if not files_exist:
                    op_keypoints = f'{output_base}/c{label}/{file_primary_name}_keypoints.pt'
                    total_images = total_images + 1
                    img = tf.io.read_file(f'{self.config.TRAIN_DATA}/c{label}/{orig_filename}')
                    img = tf.image.decode_jpeg(img)
                    pose_img, annotated, keypoints_with_scores = self._get_pose(img)
                    Path(f'{output_base}/c{label}').mkdir(parents=True, exist_ok=True)
                    tf.keras.utils.save_img(op_pose, pose_img)
                    tf.keras.utils.save_img(op_annotated, annotated)
                    torch.save(keypoints_with_scores, op_keypoints)
                    self._process_and_save_pose(op_pose, op_cropped_pose)
                    if pbar is not None:
                        pbar.update(1)

        if pbar is not None:
            pbar.close()


    def _movenet(self, input_image):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
          pixels. Note that the height/width should already be resized and match the
          expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = self.infer(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores

    def _get_pose(self, image):
        image_height, image_width, _ = image.shape
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.pose_config.INPUT_SIZE, self.pose_config.INPUT_SIZE)

        # Run model inference.
        keypoints_with_scores = self._movenet(input_image)
        # print(f'keypoints_with_scores: {keypoints_with_scores}')
        
        # Show just the pose    
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(
            display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = self._draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
        skeleton = self._draw_prediction_on_image(
            np.squeeze(np.zeros(display_image.shape), axis=0), keypoints_with_scores)
        return skeleton, output_overlay, keypoints_with_scores

    def _draw_prediction_on_image(self, 
        image, keypoints_with_scores, crop_region=None, close_figure=False,
        output_image_height=None):
        """Draws the keypoint predictions on image.

            Args:
            image: A numpy array with shape [height, width, channel] representing the
              pixel values of the input image.
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
              the keypoint coordinates and scores returned from the MoveNet model.
            crop_region: A dictionary that defines the coordinates of the bounding box
              of the crop region in normalized coordinates (see the init_crop_region
              function below for more detail). If provided, this function will also
              draw the bounding box on the image.
            output_image_height: An integer indicating the height of the output image.
              Note that the image aspect ratio will be the same as the input image.

          Returns:
            A numpy array with shape [out_height, out_width, channel] representing the
            image overlaid with keypoint predictions.
          """
        height, width, channel = image.shape
        aspect_ratio = float(width) / height
        fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
        # To remove the huge white borders
        fig.tight_layout(pad=0)
        ax.margins(0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.axis('off')

        im = ax.imshow(image)
        line_segments = LineCollection([], linewidths=(4), linestyle='solid')
        ax.add_collection(line_segments)
        # Turn off tick labels
        scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

        (keypoint_locs, keypoint_edges,
           edge_colors) = self._keypoints_and_edges_for_display(
           keypoints_with_scores, height, width)

        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
        if keypoint_edges.shape[0]:
            line_segments.set_segments(keypoint_edges)
            line_segments.set_color(edge_colors)
        if keypoint_locs.shape[0]:
            scat.set_offsets(keypoint_locs)

        if crop_region is not None:
            xmin = max(crop_region['x_min'] * width, 0.0)
            ymin = max(crop_region['y_min'] * height, 0.0)
            rec_width = min(crop_region['x_max'], 0.99) * width - xmin
            rec_height = min(crop_region['y_max'], 0.99) * height - ymin
            rect = patches.Rectangle(
                (xmin,ymin),rec_width,rec_height,
                linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
          fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        if output_image_height is not None:
            output_image_width = int(output_image_height / height * width)
            image_from_plot = cv2.resize(
                image_from_plot, dsize=(output_image_width, output_image_height),
                 interpolation=cv2.INTER_CUBIC)
        return image_from_plot

    def _keypoints_and_edges_for_display(self, keypoints_with_scores,
                                        height,
                                        width,
                                        keypoint_threshold=0.11):
        """Returns high confidence keypoints and edges for visualization.

        Args:
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
              the keypoint coordinates and scores returned from the MoveNet model.
        height: height of the image in pixels.
        width: width of the image in pixels.
            keypoint_threshold: minimum confidence score for a keypoint to be
            visualized.

        Returns:
            A (keypoints_xy, edges_xy, edge_colors) containing:
              * the coordinates of all keypoints of all detected entities;
              * the coordinates of all skeleton edges of all detected entities;
              * the colors in which the edges should be plotted.
        """
        keypoints_all = []
        keypoint_edges_all = []
        edge_colors = []
        num_instances, _, _, _ = keypoints_with_scores.shape
        for idx in range(num_instances):
            kpts_x = keypoints_with_scores[0, idx, :, 1]
            kpts_y = keypoints_with_scores[0, idx, :, 0]
            kpts_scores = keypoints_with_scores[0, idx, :, 2]
            kpts_absolute_xy = np.stack(
                [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
            kpts_above_thresh_absolute = kpts_absolute_xy[
                kpts_scores > keypoint_threshold, :]
            keypoints_all.append(kpts_above_thresh_absolute)

            for edge_pair, color in self.pose_config.KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                  kpts_scores[edge_pair[1]] > keypoint_threshold):
                    x_start = kpts_absolute_xy[edge_pair[0], 0]
                    y_start = kpts_absolute_xy[edge_pair[0], 1]
                    x_end = kpts_absolute_xy[edge_pair[1], 0]
                    y_end = kpts_absolute_xy[edge_pair[1], 1]
                    line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                    keypoint_edges_all.append(line_seg)
                    edge_colors.append(color)
        if keypoints_all:
            keypoints_xy = np.concatenate(keypoints_all, axis=0)
        else:
            keypoints_xy = np.zeros((0, 17, 2))

        if keypoint_edges_all:
            edges_xy = np.stack(keypoint_edges_all, axis=0)
        else:
            edges_xy = np.zeros((0, 2, 2))
        return keypoints_xy, edges_xy, edge_colors


    def _process_and_save_pose(self, orig_pose_file, cropped_pose_file, verbose=False):
        # Read the gray scale image.
        img = cv2.imread(orig_pose_file, cv2.IMREAD_GRAYSCALE)
        PADDING = 20
        if verbose:
            plt.title('Original (grayscale)')
            plt.imshow(img, cmap='gray')
            plt.show()

        # Quick fix for removing the white border that the pose isolation code adds.
        zero_pixels = np.argwhere(img == 0)
        border_top = np.min(zero_pixels[:,0])
        border_left  = np.min(zero_pixels[:,1])
        border_bottom = np.max(zero_pixels[:,0])
        border_right = np.max(zero_pixels[:,1])
        img = img[border_top:border_bottom, border_left:border_right]
        if verbose:
            plt.title('Border removed')
            plt.imshow(img, cmap='gray')
            plt.show()

        # Find the crop region, accounting for the white border we just removed.
        positions = np.argwhere(img > 0)
        top = np.min(positions[:,0]) + border_top
        left  = np.min(positions[:,1]) + border_left
        bottom = np.max(positions[:,0]) + border_top
        right = np.max(positions[:,1]) + border_left
        d = max(right - left, bottom - top)
        padding_x = d - (bottom - top)
        padding_y = d - (right - left)
        if verbose:
            print(f'top:{top}, left:{left}, bottom:{bottom}, right:{right}, d:{d}')
        
        # Re-read the color image and apply the crop
        img = cv2.imread(orig_pose_file)
        crop_img = img[top:bottom, left:right]
        if verbose:
            plt.title('Cropped (grayscale)')
            plt.imshow(crop_img)
            plt.show()
        
        # Add padding.
        crop_img = cv2.copyMakeBorder(crop_img, PADDING, PADDING + padding_x, PADDING, padding_y + PADDING, cv2.BORDER_CONSTANT, None, 0)
        crop_img = cv2.resize(crop_img, (256, 256), interpolation = cv2.INTER_AREA)
        if verbose:
            plt.title('Padded and resized')
            plt.imshow(crop_img)
            plt.show()
        cv2.imwrite(cropped_pose_file, crop_img)
            
    # def batch_post_processing(labels, pose_config, reduced=False, limit=10):
    #     total_images = 0
    #     start = time.process_time()

    #     for label in labels:
    #         if limit is not None:
    #             if total_images > limit:
    #                 break
    #         print(f'Processing c{label} files...')
    #         dataset = customdataset.DriverDataset(config, reduced=reduced, label=label)
    #         dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False, collate_fn=dataset.get_image_from)
    #         for batch_idx, samples in enumerate(dataloader):
    #             total_images = total_images + 1
    #             # summary = []
    #             if batch_idx % 1000 == 0:
    #                 print(f'Processed {label}:{batch_idx} files...')
    #                 print(f'Extracted from {total_images} files...')
    #             if limit is not None:
    #                 if total_images > limit:
    #                     break
    #             _, label, orig_filename = samples
    #             filename = os.path.basename(orig_filename)
    #             file_primary_name, _ = os.path.splitext(os.path.basename(filename))
    #             orig_pose_file = f'{pose_config.FEATURES_FOLDER_FULL}/c{label}/{file_primary_name}_pose.png'
    #             cropped_pose_file = f'{pose_config.FEATURES_FOLDER_FULL}/c{label}/{file_primary_name}_pose_cropped.png'
    #             # Open the file and crop it to just the pose + a border of 5 pixels and then save it.
    #             # Resize it?
    #             self._process_and_save_pose(orig_pose_file, cropped_pose_file)