import torchvision.models as models

from hyperopt import STATUS_OK,rand, tpe, Trials, fmin, hp
from hyperopt.early_stop import no_progress_loss
from torch import nn

class ResNet152(nn.Module):
    def __init__(self, progress=True):
        super(ResNet152, self).__init__()
        
        # load the pretrained model
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT, progress=progress)

        # select till the last layer
        # Dropping output layer (the ImageNet classifier)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):

      x = self.model(x)
      return x


class LogisticRegression(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.model = nn.Linear(input_dim, output_dim)

     def forward(self, x):
         outputs = torch.sigmoid(self.model(x))
         return outputs
    
# Optimizer definition
def optimize(opt_dict, best_dict, X_train, y_train, max_evals = 50, scoring_fn = 'neg_log_loss' random_state = config.SEED):
    """
        Runs hyperopt for all the models in opt_dict. Adds the best hyperparameter set for each model.
        Returns dictionary of best hyperparameter set.
    """
    # Define TPE algorithm for all optimizers
    tpe_algo = tpe.suggest

    # Iterate over opt_dict
    for k,v in opt_dict.items():
        ## Attributes
        model_name = k
        model = v['model']
        params = v['params']

        ## objective function definition
        def f(params):
            loss = None
            try:
                m = model(random_state = random_state, **params)
                loss = -cross_val_score(m, X_train, y_train, scoring = scoring_fn).mean()
            except: AttributeError

            return {'loss': loss, 'status': STATUS_OK}

        ## Define trial space
        trials = Trials()

        print(f"Optimizing {k} model...")

        ## optimize
        best = fmin(
            fn = f,
            space = params,
            algo = tpe_algo,
            max_evals = max_evals,
            trials = trials,
            early_stop_fn=no_progress_loss(1e-10)
        )

        best_dict[model_name] = best
        print(f"Best hyperparameter set for {model_name} is {best}")
        print("\n")

    return best_dict