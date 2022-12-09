import pickle
import enums

class ModelSummarizer:
    def __init__(self, config):
        self.config = config

    def load_saved_model(self, model_filename, scaler_filename):
        model_filename = f'{config.OUTPUT_FOLDER}/saved_models/{model_filename}'
        scaler_filename = f'{config.OUTPUT_FOLDER}/saved_models/{scaler_filename}'
        with open(model_filename, 'rb') as model_file, open(scaler_filename, 'rb') as scaler_file:
            return (pickle.load(model_file), pickle.load(scaler_file))

    def create_scaler(self, X, test_idx, scaler_type):
        X_train = X[train_idx, :].copy()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        pickle.dump(scaler, open(f'{config.SAVED_MODELS_FOLDER}/{scaler_type.value}','wb'))

    def evaluate_model(self, X, y, test_idx, val_idx, model_file=f'multiclass_random_hyperparam_model.pkl', scaler_file):
        model, scaler = load_saved_model(model_filename, scaler_filename)
    

