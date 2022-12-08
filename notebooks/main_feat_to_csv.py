import csv
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


list_features = list()
list_file_name = ['c0.txt', 'c1.txt', 'c2.txt', 'c3.txt', 'c4.txt', 'c5.txt', 'c6.txt', 'c7.txt', 'c8.txt', 'c9.txt']

# Merge features from all the classes
for file_name in list_file_name:
    with open(file_name, 'rb') as f:
        pkl_obj = pickle.load(f)
        for key in pkl_obj.keys():
            this_feature = [file_name[0:2]] + [key] + pkl_obj[key]
            list_features.append(this_feature)


# Write into a CSV file
csv_file_name = 'body_parts_feat.csv'
fields = ['Class', 'ImageFile', 'LeftCenterX', 'LeftCenterY', 'LeftOrient', 'RightCenterX', 'RightCenterY', 'RightOrient']
df_dict = {
    'Class': [feature[0] for feature in list_features],
    'ImageFile': [feature[1] for feature in list_features],
    'LeftCenterX': [feature[2] for feature in list_features],
    'LeftCenterY': [feature[3] for feature in list_features],
    'LeftOrient': [feature[4] for feature in list_features],
    'RightCenterX': [feature[5] for feature in list_features],
    'RightCenterY': [feature[6] for feature in list_features],
    'RightOrient': [feature[7] for feature in list_features],
}
df = pd.DataFrame.from_dict(df_dict)
# df.to_csv(csv_file_name, index=False)


# Get box plots information for left arm-orientation
df_without_na = df.dropna()
class_tags = df['Class'].unique()
list_left_arm_orients = list()
for this_tag in class_tags:
    left_arm_orients = df_without_na.loc[df['Class'] == this_tag]['LeftOrient'].tolist()
    list_left_arm_orients.append(left_arm_orients)
# Plot box-plot for right arm
ax = sns.boxplot(list_left_arm_orients)
ax.set_xticklabels(class_tags.tolist())
ax.set(xlabel='Class tags', ylabel='Orientation (in rad)', title='Left arm orientations across different classes')
plt.show()


# Get box plots information for right arm-orientation
df_without_na = df.dropna()
class_tags = df['Class'].unique()
list_right_arm_orients = list()
for this_tag in class_tags:
    right_arm_orients = df_without_na.loc[df['Class'] == this_tag]['RightOrient'].tolist()
    list_right_arm_orients.append(right_arm_orients)
# Plot box-plot for right arm
ax = sns.boxplot(list_right_arm_orients)
ax.set_xticklabels(class_tags.tolist())
ax.set(xlabel='Class tags', ylabel='Orientation (in rad)', title='Right arm orientations across different classes')
plt.show()

print('here')
