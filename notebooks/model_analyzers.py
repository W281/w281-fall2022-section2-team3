import pickle
import enums
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


class ModelSummarizer:
    def __init__(self, config):
        self.config = config

    def create_scaler(self, X, test_idx, scaler_type):
        X_train = X[train_idx, :].copy()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        pickle.dump(scaler, open(f'{self.config.SAVED_MODELS_FOLDER}/{scaler_type.value}','wb'))

    def _check_against_ds(self, X, y, model_name, feature_name, dataset_name, model_filename, scaler_file_name, pca_model_file_name=None, best_params=False):
        out_file = f'confusion_matrix_{model_name}_{feature_name}_{dataset_name}.jpg'

        if pca_model_file_name is not None:
            with open(f'{self.config.SAVED_MODELS_FOLDER}/{pca_model_file_name}', 'rb') as pca_modelfile:
                pca_model =  pickle.load(pca_modelfile)
                X_val = pca_model.transform(X)
        else:
            X_val = X

        if scaler_file_name is not None:
            with open(f'{self.config.SAVED_MODELS_FOLDER}/{scaler_file_name}', 'rb') as scalerfile:
                scaler = pickle.load(scalerfile)
                X_val = scaler.transform(X_val)
                
            
        with open(f'{self.config.SAVED_MODELS_FOLDER}/{model_filename}', 'rb') as modelfile:
            model = pickle.load(modelfile)
            y_pred = model.predict(X_val)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f'Confusion Matrx')
            disp_labels = [self.config.class_dict[key] for key in sorted(self.config.class_dict.keys())]
            cm = metrics.confusion_matrix(y, y_pred, labels=model.classes_)
            
            print(f'Model Summary:',)
            print(model)
            if best_params:
                print(f'Best Parameters:',)
                print(model.best_params_)

            result1 = metrics.classification_report(y, y_pred)
            print(f'Model Classification Report:',)
            print (result1)
            accuracy_score = metrics.accuracy_score(y, y_pred)
            print(f'Model Accuracy:',accuracy_score)
            
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
            disp.plot(xticks_rotation='30', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')        
            plt.subplots_adjust(bottom=0.3)
            plt.title('Confusion Matrix')
            plt.savefig(f'{self.config.OUTPUT_FOLDER}/report_plots/{out_file}', dpi=300)
            plt.tight_layout()        
            plt.show()
            return y_pred, accuracy_score

    def evaluate_models_against(self, dataset_name, X, y):
        # Check how KNN model with with principal components do against the dataset.
        print(f'KNN Using Keypoints PCA against {dataset_name} Dataset')
        print('========================================================================================')
        knn_pca16_y_pred, knn_pca16_acc = self._check_against_ds(X.copy(),
                         y.copy(), dataset_name=dataset_name,
                         model_name='knn', feature_name='keypoints_pca',
                         model_filename = 'knn_keypoints_pca.pkl',
                         scaler_file_name = None,
                         pca_model_file_name='pca_for_knn_keypoints_pca.pkl')


        # Check how logistic regression model with principal components does against the dataset.
        print(f'Logistic Regression Using Keypoints PCA against {dataset_name} Dataset')
        print('========================================================================================')
        lr_pca16_y_pred, lr_pca16_acc = self._check_against_ds(X.copy(),
                         y.copy(), dataset_name=dataset_name,
                         model_name='logistic', feature_name='keypoints_pca',
                         model_filename = 'model_logistic_keypoints_pca.pkl',
                         scaler_file_name = 'scaler_logistic_keypoints_pca.pkl',
                         pca_model_file_name='model_pca_16_keypoints_pca.pkl',
                         best_params=True)

        print(f'Logistic Regression Using Keypoints against {dataset_name} Dataset')
        print('========================================================================================')
        # Check how logistic regression model with keypoints features does against the dataset.
        lr_keypoints_y_pred, lr_keypoints_acc = self._check_against_ds(X.copy(), 
                        y.copy(), dataset_name=dataset_name,
                        model_name='logistic', feature_name='keypoints',
                        model_filename = 'model_logistic_keypoints.pkl',
                        scaler_file_name = 'scaler_logistic_keypoints.pkl',
                        best_params=True)
        
        print(f'Random Forest Using Keypoints against {dataset_name} Dataset')
        print('========================================================================================')
        # Check how randomg forest model with keypoints features does against the dataset.
        rf_y_pred, rf_acc = self._check_against_ds(X.copy(), 
                        y.copy(), dataset_name=dataset_name,
                        model_name='randomforest', feature_name='keypoints',
                        model_filename = 'model_rf_keypoints.pkl',
                        scaler_file_name = 'scaler_logistic_keypoints.pkl')
        return {
            'knn_pca16': (knn_pca16_y_pred, knn_pca16_acc),
            'lr_pca16': (lr_pca16_y_pred, lr_pca16_acc),
            'lr_keypoints': (lr_keypoints_y_pred, lr_keypoints_acc),
            'rf_keypoints': (rf_y_pred, rf_acc)
        }

    def generate_sumamry(self):
        summary_cols = ['name', 'model_family', 'features_used', 'parameters', 'validation_accuracy', 'holdout_accuracy', 'desc']
        summary = []
        knn_summary = ['KNN-PCA', 'KNN', '16 Principal Components of Pose', 'N=3', 0.95, 0.89, '']
        lr_pca_summary = ['LR-PCA', 'LogisticRegression', '16 Principal Components of Pose', 'regularization: l1; solver: saga; loss:negative log loss; HyperParmam Tuning: Random grid search; C: 9.5', 0.79, 0.8, '']
        lr_summary = ['LR', 'LogisticRegression', '12 keypoints and 2 hand angles', 'regularization: l1; solver: saga; loss:negative log loss; HyperParmam Tuning: Random grid search; ', 0.85, 0.83, '']
        fr_summary = ['RF', 'RandomForest', '12 keypoints and 2 hand angles', 'max_depth=10, n_estimators=173; HyperParmam Tuning: Hyperopt; ', 0.94, 0.88, '']
