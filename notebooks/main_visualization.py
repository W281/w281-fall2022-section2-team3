# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Feature dataframe
csv_file_name = 'body_parts_feat.csv'
df = pd.read_csv(csv_file_name)
df_without_na = df.dropna()

while True:
    # Random image index
    random_idx = np.random.choice(10000, 1)
    this_feature = df_without_na.iloc[random_idx]
    source_image_path = 'imgs/train/' + this_feature['Class'].item() + '/' + this_feature['ImageFile'].item()
    source_image = plt.imread(source_image_path)

    # Display image
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)
    ax.imshow(source_image)

    # Display left arm orientation
    center_x_left, center_y_left, orient_left = this_feature['LeftCenterX'].item(), -this_feature['LeftCenterY'].item(), this_feature['LeftOrient'].item()
    orient_x_left, orient_y_left = center_x_left + 100*np.cos(orient_left), center_y_left - 100*np.sin(orient_left)
    ax.plot(center_x_left, center_y_left, 'r*')
    ax.annotate("", xy=(orient_x_left, orient_y_left), xytext=(center_x_left, center_y_left),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=3))

    # Display right arm orientation
    center_x_right, center_y_right, orient_right = this_feature['RightCenterX'].item(), -this_feature['RightCenterY'].item(), this_feature['RightOrient'].item()
    orient_x_right, orient_y_right = center_x_right + 100 * np.cos(orient_right), center_y_right - 100 * np.sin(orient_right)
    ax.plot(center_x_right, center_y_right, 'r*')
    ax.annotate("", xy=(orient_x_right, orient_y_right), xytext=(center_x_right, center_y_right),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=3))

    ax.set(title='Detected arms with orientations')
    plt.show()
