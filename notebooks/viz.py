import numpy as np
from matplotlib import pyplot as plt
import glob
import random
import time
import os
import pandas as pd
import seaborn as sns
import customdataset
import enums
from torch.utils.data import DataLoader
import feature_helpers
import matplotlib.ticker as ticker


class Vizualizer:
    def __init__(self, config, face_config, pose_config, tqdm=None):
        self.config = config
        self.face_config = face_config
        self.pose_config = pose_config
        self.tqdm = tqdm

    def infer_and_plot(self, model, infer, title, folder_name, num_images=9, cols=3, plt_width=None, plt_height=None, out_file=None, add_border=True, title_color='white'):
        files = list(glob.glob(folder_name + '/*.jpg'))
        rows = num_images // cols
        rows = rows if rows * cols >= num_images else rows + 1

        if plt_width and plt_height:
            fig, axes = plt.subplots(rows, cols, figsize=(plt_width, plt_height))
        else:
            fig, axes = plt.subplots(rows, cols)
        if add_border:
            rect = plt.Rectangle(
                # (lower-left corner), width, height
                (0.00, 0.0), 1., 1., fill=False, color="k", lw=2, 
                zorder=1000, transform=fig.transFigure, figure=fig
            )
            fig.patches.extend([rect])
        axes = np.array(axes).flatten().tolist()
        times = []
        for i in range(num_images):
            ax = axes[i]
            filename = random.choice(files)
            times.append(infer(ax, model, filename))
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.axis('off')
        fig.suptitle(title, fontsize=24, color=title_color)

        plt.tight_layout(pad=.05, h_pad=None, w_pad=None, rect=None)
        if out_file is not None:
            plt.savefig(out_file, bbox_inches='tight')
        plt.show()
        return times

    def plot_raw_class_counts(self, out_file='class_distribution.jpg'):
        classes = list(self.config.class_dict.keys())
        classes.sort()
        d = {"img" : [], "class" : []}
        for c in classes:
            imgs = [img for img in os.listdir(os.path.join(self.config.TRAIN_DATA, f'c{c}')) if not img.startswith(".")]
            for img in imgs:
                d["img"].append(img)
                d["class"].append(self.config.class_dict.get(c))
        df = pd.DataFrame(d)
        ax = sns.countplot(data=df, y="class", palette='Set2')
        ax.set(title="Distribution of Training Data")
        ax.set_xlabel('Count')
        ax.set_ylabel('Class')
        # ax.tick_params(axis='x', rotation=90)
        plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300, bbox_inches = "tight")
        plt.tight_layout()
        plt.show()


    def plot_samples(self, num=5, out_file='class_samples.jpg'):
        def sample(c, num):
            # TODO: Do random sampling instead of the first num images.
            d = [os.path.join(self.config.TRAIN_DATA, f'c{c}', img) for img in os.listdir(os.path.join(self.config.TRAIN_DATA, f'c{c}')) if not img.startswith(".")]
            return d[0:num], self.config.class_dict.get(c)
        # load images per class
        # Display them in each row.
        fig, axes = plt.subplots(10, num, figsize=[21, 42])
        pad = 5 # in points
        classes = list(self.config.class_dict.keys())

        for row, class_code in enumerate(classes):
            imgs, class_name = sample(class_code, num)      
            for i, img_file in enumerate(imgs):
                img = plt.imread(img_file)
                axis = axes[row][i]
                if i == 0:
                    axis.annotate(class_name, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=24, ha='center', va='baseline')
                axis.imshow(img)
                axis.set_xticks([])
                axis.set_yticks([])

        #   fig.suptitle('Samples From Each Class', fontsize=32, y=0.99)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300, bbox_inches = "tight")
        plt.show()

    def plot_faces_summary(self, out_file='faces_distribution.jpg'):
        summary_csv = f'{self.face_config.FEATURES_FOLDER}/{self.face_config.FACE_SUMMARY_NAME}'
        df_summary = pd.read_csv(summary_csv)
        df_summary['class_name'] = df_summary['class'].map(self.config.class_dict)
        
        df_0 = df_summary[df_summary['num_faces'] == 0].groupby('class_name').agg(
            no_faces=pd.NamedAgg(column="filename", aggfunc="count")
        )
        df_1 = df_summary[df_summary['num_faces'] == 1].groupby('class_name').agg(
            one_face=pd.NamedAgg(column="filename", aggfunc="count")
        )
        df_gt_1 = df_summary[df_summary['num_faces'] > 0].groupby('class_name').agg(
            many_faces=pd.NamedAgg(column="filename", aggfunc="count")
        )
        
        pd_merged = pd.merge(pd.merge(df_0, df_1, on='class_name'), df_gt_1, on='class_name')
        pd_merged.plot(title='Face Identification Summary', figsize=(8, 6), kind='bar', 
                       ylabel='', xlabel='', rot=45, grid=False)
        plt.tight_layout()
        plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300)
        plt.show()

    def display_faces(self, per_group_count=3, out_file='faces.jpg'):
        def get_images(sample_type):
            dataset = customdataset.MainDataset(self.config, self.face_config, self.pose_config,
                                                sample_type=sample_type,
                                                image_types=[enums.ImageTypes.FACE_ANNOTATED])
            
            dataloader = DataLoader(dataset, num_workers=0, batch_size=1,
                                    shuffle=True, collate_fn=dataset.get_image_from)
            sampled = []
            for i, sample in enumerate(dataloader):
                if i >= per_group_count:
                    break
                images, label, filename = sample
                image = images[enums.ImageTypes.FACE_ANNOTATED]
                sampled.append(image)
            return sampled
        
        fig, axes = plt.subplots(per_group_count, 3, figsize=[18, 15], dpi=72)
        sampled = list(zip(get_images(enums.SampleType.WITH_NO_FACE),
                           get_images(enums.SampleType.WITH_JUST_ONE_FACE), 
                           get_images(enums.SampleType.WITH_MORE_THAN_ONE_FACE)))
        titles = ['No Face Identified', 'Exactly One Face Identified', 'Multiple Faces Identified']
        for i in range(per_group_count):
            for j, image in enumerate(sampled[i]):
                if i == 0:
                    axes[i][j].set_title(titles[j], fontsize = 20)
                axes[i][j].imshow(image)
                axes[i][j].axis('off')
        plt.suptitle('Random Samples Annotated With Faces Identified', fontsize = 26)
        plt.tight_layout()
        plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300)    
        plt.show()

    def display_poses(self, rows=3, out_file='poses.jpg'):
        dataset = customdataset.MainDataset(self.config, self.face_config, self.pose_config,
                                            sample_type=enums.SampleType.TRAIN_VALIDATION,
                                            image_types=[enums.ImageTypes.ORIGINAL, 
                                            enums.ImageTypes.POSE_ANNOTATED, enums.ImageTypes.POSE])

        dataloader = DataLoader(dataset, num_workers=0, batch_size=1,
                                shuffle=True, collate_fn=dataset.get_image_from)
        fig, axes = plt.subplots(rows, 3, figsize=[18, 15], dpi=72)
        axes[0][0].set_title('Original', fontsize = 20)
        axes[0][1].set_title('Annotated With Pose', fontsize = 20)
        axes[0][2].set_title('Pose Extracted', fontsize = 20)

        for i, sample in enumerate(dataloader):
            if i >= rows:
                break
            images, label, filename = sample
            axes[i][0].imshow(images[enums.ImageTypes.ORIGINAL])
            axes[i][0].axis('off')
            axes[i][1].imshow(images[enums.ImageTypes.POSE_ANNOTATED])
            axes[i][1].axis('off')
            axes[i][2].imshow(images[enums.ImageTypes.POSE])
            axes[i][2].axis('off')
        plt.suptitle('Random Samples Annotated With Poses', fontsize = 26)
        plt.tight_layout()
        plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300)
        plt.show()

    def plot_features(self, included_labels=None, count_per_label = 1, out_file='visual_features.jpg'):
        included_labels = self.config.class_dict.keys() if included_labels is None else included_labels
        def display_img(ax_idx, img, row_name, col_name, first_row, first_col):
            ax = axes[ax_idx]
            ax.imshow(img, cmap='gray')
            if first_row:
                ax.set_title(col_name, fontsize=32)
            if first_col:
                ax.set_ylabel(row_name, fontsize = 32)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            
        # Load the data
        image_types = [
            enums.ImageTypes.FACE_ANNOTATED,
            enums.ImageTypes.POSE_ANNOTATED,
            enums.ImageTypes.ORIGINAL,
            enums.ImageTypes.POSE, 
            enums.ImageTypes.FACE
        ]
        feature_extractor = feature_helpers.FeatureExtractor(self.config, self.face_config, self.pose_config, self.tqdm)
        data = feature_extractor.load_data(image_types=image_types, shuffle=True, 
                                           sample_type=enums.SampleType.TRAIN_VALIDATION,
                                           count_per_label=count_per_label, include_feature_vectors=True)
        _, hogs = feature_extractor.get_hog_features(data[enums.DataColumn.FACE.value])
        data['hog_img'] = hogs
        _, cannies = feature_extractor.get_canny_features(data[enums.DataColumn.FACE.value])
        data['canny_img'] = cannies
        
        fig, axes = plt.subplots(nrows=len(included_labels), ncols=len(image_types) + 2, figsize=[36, 48], dpi=72)
        axes = axes.flatten()
        first_row = True
        pbar = None
        pbar = None if self.tqdm is None else self.tqdm(unit='images', desc=f'Loading image', total=len(included_labels)*len(image_types) + 2)

        for i, cur_label in enumerate(included_labels):
            label_df = data[data['label'] == cur_label]
            for index, row in label_df.iterrows():
                cur_filename = row[enums.DataColumn.FILENAME.value]
                face_annotated_img = row[enums.DataColumn.FACE_ANNOTATED.value]
                pose_annotated_img = row[enums.DataColumn.POSE_ANNOTATED.value]
                original_img = row[enums.DataColumn.ORIGINAL.value]
                pose_img = row[enums.DataColumn.POSE.value]
                face_img = row[enums.DataColumn.FACE.value]
                hog = row['hog_img']
                canny = row['canny_img']

                imgs_to_display = [
                    ('Original', original_img),
                    ('Face Annotated', face_annotated_img),
                    ('Face Extracted', face_img),
                    ('Face - Canny', canny),
                    ('Face - Hog', hog),
                    ('Pose Annotated', pose_annotated_img),
                    ('Pose Extracted', pose_img)
                ]
                images_per_label = len(imgs_to_display)
                first_col = True
                for j, item in enumerate(imgs_to_display):
                    col_name, img = item
                    row_name = self.config.class_dict[cur_label]
                    display_img(i * images_per_label + j, img, row_name, col_name, first_row, first_col)
                    if pbar is not None:
                        pbar.update(1)
                    first_col = False
                first_row = False

        if pbar is not None:
            pbar.close()
        plt.suptitle('Visualizing The Features', fontsize=48)
        plt.tight_layout(pad=4, h_pad=None, w_pad=None, rect=None)
        plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300)
        plt.show()

    def show_scores_distributions(self, out_file='pose_scores.jpg'):
        feature_extractor = feature_helpers.FeatureExtractor(self.config, self.face_config, self.pose_config, self.tqdm)
        df = feature_extractor.keypoints_relative_to_nose(sample_type=enums.SampleType.TRAIN_VALIDATION)
        
        print(f'{df[df["nose_score"] < self.pose_config.KEYPOINT_THRESHOLD].shape[0]} images are missing noses')
        print(f'Nose: min score: {df["nose_score"].min()}, max score: {df["nose_score"].max()}')
        
        score_cols = [f'{name}_score' for name in self.pose_config.SORTED_KEYPOINT_NAMES]
        df[score_cols].plot(kind='hist', figsize=(10, 10), layout=(5,4), subplots=True, 
                            sharey=True, title='Pose Keypoints Score Summary')
        plt.tight_layout()
        plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300)
        plt.show()

    def plot_keypoints_relative_positions(self, out_file='wrists_offset_distribution.jpg'):
        feature_extractor = feature_helpers.FeatureExtractor(self.config, self.face_config, self.pose_config, self.tqdm)
        df = feature_extractor.keypoints_relative_to_nose(sample_type=enums.SampleType.TRAIN_VALIDATION)
        df['label_name'] = df['label'].map(self.config.class_dict)
        
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[18, 12], dpi=72)
            
        row = 0
        col = 0
        ax = axes[row][col]
        ax.set_title('Right Wrist: X Offset')
        sns.swarmplot(ax=ax, y = df['right_wrist_offset_x'], x = df['label_name'], s=1)

        row = 0
        col = 1
        ax = axes[row][col]
        ax.set_title('Right Wrist: y Offset')
        sns.swarmplot(ax=ax, y = df['right_wrist_offset_y'], x = df['label_name'], s=1)

        row = 1
        col = 0
        ax = axes[row][col]
        ax.set_title('Left Wrist: x Offset')
        sns.swarmplot(ax=ax, y = df['left_wrist_offset_x'], x = df['label_name'], s=1)

        row = 1
        col = 1
        ax = axes[row][col]
        ax.set_title('Left Wrist: y Offset')
        sns.swarmplot(ax=ax, y = df['left_wrist_offset_y'], x = df['label_name'], s=1)

        for ax in axes.flatten():
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        plt.suptitle('Offset Distribution For Wrists')
        plt.tight_layout()
        plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300)
        plt.show()

    # Learning curve plotting method is from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        scoring=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
    ):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        scoring : str or callable, default=None
            A str (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
        )
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(
            fit_time_sorted,
            test_scores_mean_sorted - test_scores_std_sorted,
            test_scores_mean_sorted + test_scores_std_sorted,
            alpha=0.1,
        )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt
