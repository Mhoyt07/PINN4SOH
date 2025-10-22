import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils.util import AverageMeter,get_logger,eval_metrix
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# Custom sine activiation layer for neural network.]
class Sin(nn.Module):
    '''
    Custom sine activation layer for the neural network.

    This layer applies the sine function element-wise to its input tensor.
    It introduces smooth, periodic nonlinearity into the model, which can
    be useful in physics-informed neural networks (PINNs) or problems
    involving oscillatory or periodic patterns.

    Example:
        x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        sin_layer = Sin()
        y = sin_layer(x)
        # y = [[sin(0.0), sin(1.0)], [sin(2.0), sin(3.0)]]
    '''
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) model with sine activations and dropout.

    This network is a fully connected feedforward neural network.
    It maps an input tensor of size `input_dim` to an output tensor
    of size `output_dim` through several hidden layers.

    Each hidden layer uses:
      - Linear transformation (weights + bias)
      - Sin() activation for smooth nonlinearity
      - Optional Dropout for regularization

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        layers_num (int): Total number of linear layers (≥ 2).
        hidden_dim (int): Number of neurons per hidden layer.
        droupout (float): Dropout probability (0–1).

    Example:
        model = MLP(input_dim=17, output_dim=1, layers_num=4, hidden_dim=50)
        x = torch.randn(10, 17)
        y = model(x)
        # y shape → (10, 1)
    """
    def __init__(self,input_dim=17,output_dim=1,layers_num=4,hidden_dim=50,droupout=0.2):
        super(MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        # Create a list that contains all of our neural network layers
        self.layers = []
        for i in range(layers_num):
            if i == 0:
                # The first to layers are a linear trnasforamtion that goes from input_dim number of variables to hidden_dim number of layers
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                # They next go through a sign layer to introduce nonlinearity
                self.layers.append(Sin())
            elif i == layers_num-1:
                # Now if the layer is the last layer, then it goes form the hidden_dim number of neurons to the output_dim number of neurons
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                # Every layer that is not the first one (two: linear transform and sin) is a layer with 3 steps(so three layers):
                # The linear layer form hidden_dim neurso to hidden_dim neurons
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                # A sin layer to introduce nonlinearity
                self.layers.append(Sin())
                # And a droupout layer to regularize the model and avoid overfitting
                self.layers.append(nn.Dropout(p=droupout))
        # Self.net takes the layers that we have and creates a pipeline for the inputs to flow through
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        """
        Initialize the weights of all Linear layers in the network using Xavier normal initialization.
        Sin() and Dropout layers are not initialized because they have no learnable parameters.
        """
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self,x):
        """
        Runs input tensor through given pipeline to produce an output.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        """
        x = self.net(x)
        return x


class Predictor(nn.Module):
    """
    Prediction head network that maps input features to a single output (predicted SOH).

    Structure:
    - Dropout(p=0.2): randomly drops 20% of input features during training to prevent overfitting.
    - Linear(input_dim → 32): fully connected layer to reduce/input features to 32-dimensional hidden representation.
    - Sin(): sine activation applied element-wise to introduce non-linearity.
    - Linear(32 → 1): final layer that outputs a single predicted SOH value.

    Methods:
    - forward(x): passes input through the network and returns the predicted SOH.
    """
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        # Creates a pipeline of layers that the input will flow through
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,1)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)

class Solution_u(nn.Module):
    """
    Solution neural network for predicting battery SOH (state of health).

    Structure:
    - Encoder: MLP that maps input features (17-dim) to a 32-dimensional embedding.
    - Predictor: Small network that takes the embedding and outputs a single scalar (predicted SOH).
    
    Methods:
    - forward(x): Passes input through encoder and predictor to produce SOH prediction.
    - get_embedding(x): Returns the 32-dimensional embedding from the encoder (useful for feature analysis).
    - _init_(): Initializes all Linear and Conv1d layers using Xavier normal for weights and zeros for biases.
    """
    def __init__(self):
        super(Solution_u, self).__init__()
        self.encoder = MLP(input_dim=17,output_dim=32,layers_num=3,hidden_dim=60,droupout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


class LR_Scheduler(object):
    """
    Learning Rate Scheduler for training the neural network.

    This class manages the learning rate (LR) for the optimizer during training,
    implementing a combination of:

    1. Warmup Phase:
    - During the first 'warmup_epochs', the LR gradually increases from 'warmup_lr' 
        to 'base_lr' over 'warmup_iter' iterations.
    - This helps prevent unstable updates at the start of training when weights are random.
    - The schedule is created using a linear interpolation via np.linspace.

    2. Cosine Decay Phase:
    - After warmup, the LR gradually decreases from 'base_lr' to 'final_lr' over
        the remaining iterations ('decay_iter') using a cosine function.
    - Cosine decay allows for smooth reduction in LR, helping the network converge 
        to a stable minimum.

    3. Parameter Group Control:
    - If 'constant_predictor_lr' is True, parameter groups named 'predictor' 
        maintain a fixed learning rate ('base_lr') throughout training.
    - Other parameter groups follow the computed warmup + cosine decay schedule.

    Key Methods:
    - step(): Updates the LR of each parameter group based on the current iteration.
            Increments the iteration counter and returns the LR for this step.
    - get_lr(): Returns the current learning rate being used.

    Usage:
    - Instantiate with the optimizer and hyperparameters.
    - Call step() once per training iteration to update learning rates.
    """
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        # Total number of warmup iterations is the iterations per eopch multiplied by the number of warmup epochs the user wants
        warmup_iter = iter_per_epoch * warmup_epochs
        # Creates an array of learning rates that lineraly increases from warmup_lr to base_lr over the warmup iterations
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        # Total number of iteratons after warmup during training
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        # Cosine decy which starts from the base_lr and slowly decreases to the final_lr at a cosine rate
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        # The complete learning rate schedule is the warmup schedule followed by the cosine decay schedule
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        """
        Update the learning rate for the current iteration.
        - Uses precomputed warmup + cosine decay schedule.
        - Can keep specific parameter groups (like predictor) constant if needed.
        - Increments the iteration counter.
        """
        for param_group in self.optimizer.param_groups:
            # If user wants predictor learning rate to be constant
            # If param group is predictor and constant predictor lr is true
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                # Sets the param groups learning rate to be the current learning rate of the schedule
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr



class PINN(nn.Module):
    def __init__(self,args):
        super(PINN, self).__init__()
        self.args = args
        # If user specified a folder to models to and it does not exist yet, creat it
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        # If there is no folder to save to, then args are saved in log_dir directiory
        # Else, they are saved in the save_folder/log_dir directory
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        # Initializes a logger object to record messages.
        self.logger = get_logger(log_dir)
        # Calls method to log all hyperparamters
        self._save_args()

        # Creates a solution u instance and puts it to the correct computing device
        self.solution_u = Solution_u().to(device)
        # Creates the dynamical_F network to model battery degradation dynamics (PDE term) and moves it to the computing device
        self.dynamical_F = MLP(input_dim=35,output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               droupout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        # Optimizer for solution_u network (learns the predicted SOH)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        # Optimizer for dynamical_F network (learns the PDE dynamics)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        # Initialize the learning rate scheduler for solution_u optimizer.
        # Handles warmup and cosine decay of learning rate over training iterations.
        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        # Mean Squared Error loss function for data fitting
        self.loss_func = nn.MSELoss()
        # ReLU activation to enforce physics constraints (e.g., u2 - u1 < 0)
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # Loss weights for combining data, PDE, and physics losses
        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    # Log all hyperparameters and arguments for reproducibility
    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    # Clears teh logger so future run starts clean
    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    # Loads model from saved location 
    def load_model(self, model_path):
        """
        Load the model weights from a saved checkpoint file.

        This method restores the saved state dictionaries for the submodules
        `solution_u` and `dynamical_F` from the given checkpoint file. It also
        ensures that all parameters in `solution_u` are set to be trainable,
        allowing further training or fine-tuning.

        Args:
            model_path (str): Path to the saved checkpoint file containing
                            model state dictionaries.
        """
        # Loads checkpoint dadta from model_path
        checkpoint = torch.load(model_path)
        # Gives solution_u and dynamical_F their saved weights
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        # Ensures every solution_u parameter is trainable
        for param in self.solution_u.parameters():
            param.requires_grad = True

    
    def predict(self,xt):
        """
        Generate predictions from the model for the given input features.

        Args:
            xt (torch.Tensor): Input tensor of shape (batch_size, input_dim), containing
                            the features for which predictions are needed.

        Returns:
            torch.Tensor: Predicted output tensor of shape (batch_size, 1), representing
                        the model’s predicted SOH (state of health) values.
        
        Description:
            This method feeds the input features through the `solution_u` network,
            which internally consists of an encoder (MLP) and a predictor head.
            It is primarily used for evaluation or inference, without computing gradients.
        """
        return self.solution_u(xt)

    
    def Test(self,testloader):
        """
        Evaluate the model on a test dataset and collect predictions and true labels.

        Args:
            testloader (torch.utils.data.DataLoader): DataLoader providing batches of test data.
                Each batch should return a tuple containing input features and labels.

        Returns:
            tuple:
                - true_label (np.ndarray): Concatenated ground-truth labels for the entire test set.
                - pred_label (np.ndarray): Concatenated predicted values for the entire test set.
        
        Description:
            This method:
            1. Puts the model in evaluation mode (disables dropout, batchnorm updates, etc.).
            2. Iterates over the test data without computing gradients.
            3. Passes each batch through the model using `predict()`.
            4. Collects true labels and predictions, converts them to NumPy arrays, and concatenates them.
            5. Returns the full arrays for later evaluation (e.g., computing MSE, MAE, or plotting).
        """
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        """
        Evaluate the model on a validation dataset and compute the mean squared error (MSE).

        Args:
            validloader (torch.utils.data.DataLoader): DataLoader providing batches of validation data.
                Each batch should return a tuple containing input features and labels.
                The format is expected to be (x1, _, y1, _), where only x1 and y1 are used.

        Returns:
            float: Mean Squared Error (MSE) between the model’s predictions and true labels
                across the entire validation set.

        Description:
            This method:
            1. Puts the model in evaluation mode to disable dropout and batch normalization updates.
            2. Iterates over the validation data without computing gradients (torch.no_grad()).
            3. Uses the `predict()` method to generate model outputs for each batch.
            4. Collects and concatenates predictions and true labels across all batches.
            5. Computes the mean squared error between predictions and true labels.
            6. Returns a single scalar MSE value that can be used to monitor validation performance
            during training or for early stopping decisions.
        """
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()

    def forward(self,xt):
        """
        Forward pass for the PINN, computing both the predicted output and the PDE residual.

        This method performs the following steps:
        1. Splits the input tensor `xt` into features `x` and time `t`.
        2. Passes `[x, t]` through the `solution_u` network to predict `u` (state of health, SOH).
        3. Computes the derivatives of `u` w.r.t. `t` (`u_t`) and `x` (`u_x`) for physics constraints.
        4. Passes `[xt, u, u_x, u_t]` through the `dynamical_F` network to obtain the PDE term `F`.
        5. Computes the PDE residual `f = u_t - F`, which quantifies how much the prediction violates the PDE.

        Args:
            xt (torch.Tensor): Input tensor of shape (batch_size, n_features+1),
                            where the last column represents time `t`.

        Returns:
            tuple:
                u (torch.Tensor): Predicted state variable (SOH) of shape (batch_size, 1).
                f (torch.Tensor): PDE residual tensor of shape (batch_size, 1),
                                used for physics-informed loss calculation.
        """
        # Enables gradients for input
        xt.requires_grad = True
        # Sets x to all features except the last column
        x = xt[:,0:-1]
        # Sets t to be the last column (time feature)
        t = xt[:,-1:]

        # Concatenates x and t and passes through solution_u to get predicted SOH
        u = self.solution_u(torch.cat((x,t),dim=1))

        # Computes gradients of u w.r.t. t and x
        # u_t: time derivative of u
        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        # u_x: spatial derivative of u
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        # Computes the PDE term F using dynamical_F network
        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        # f is the residual of the PDE, i.e., how much the network violates the physia law
        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        """
        Train the PINN model for one full epoch over the provided data.

        This method performs the following steps for each batch in the dataloader:
            1. Moves the input and target tensors to the correct device (CPU/GPU).
            2. Computes the model predictions and PDE residuals using `self.forward()`.
            3. Computes three components of the loss:
                - loss1: data loss (mean squared error between predicted and true SOH)
                - loss2: PDE loss (mean squared error of the residual f = u_t - F)
                - loss3: physics loss (enforces u2 - u1 < 0 constraint)
            4. Combines the losses using the weights `self.alpha` and `self.beta`.
            5. Performs backpropagation using `loss.backward()`.
            6. Updates the parameters of `solution_u` and `dynamical_F` using their respective optimizers.
            7. Updates the running average of each loss using `AverageMeter` for monitoring.

        Args:
            epoch (int): The current epoch number (used for logging).
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of
                (x1, x2, y1, y2) tuples for training.

        Returns:
            tuple: A tuple of three floats:
                - avg_data_loss: average data loss (loss1) over the epoch
                - avg_pde_loss: average PDE loss (loss2) over the epoch
                - avg_phys_loss: average physics loss (loss3) over the epoch

        Notes:
            - The model is set to training mode with `self.train()` to enable dropout.
            - Gradients are zeroed before each backward pass to avoid accumulation.
            - Logging occurs every 50 iterations to monitor progress.
        """
        # Puts model in training mode (to enable dropout, batchnorm updates, etc.)
        self.train()
        # Tracks running averages of each loss component
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()

        # Iterates over batches from the dataloader
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            # Moves them to correct device
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)

            # Forward pass for both time steps calculating predicted SOH and PDE residual
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            # MSE equally weighing x1 and x2
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            # The loss is MSE of residuals vs 0
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            # enforces physical rules via ReLU, meaning penalizes if u2 - u1 >0
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            # Alpha and beta are the wieghts for PDE and physics losses
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            # Backpropagation and optimization step
            # Resets gradients to 0s
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            # Computes gradients stepping backwards thorugh nodes
            loss.backward()
            # Optimizers use the computed gradients to update model weights
            # Updates all the weights at once using the lr
            self.optimizer1.step()
            self.optimizer2.step()

            # Adds new losses to the loss meters
            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

            # Adds logs to batch every 50 iterations
            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))

        # Returns the average losses for the epoch
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        """
        Train the PINN model over multiple epochs, optionally performing validation and testing,
        and saving the best model based on validation performance.

        This method performs the following steps:
            1. Iterates over the specified number of epochs (`self.args.epochs`).
            2. Calls `train_one_epoch` to train on all batches of the `trainloader` and computes
            average data, PDE, and physics losses for the epoch.
            3. Updates the learning rate using the scheduler (`self.scheduler.step()`).
            4. Logs training information (epoch, learning rate, and weighted total loss).
            5. If a validation loader is provided, computes validation MSE using `Valid()` and logs it.
            6. If the current validation MSE is lower than the best seen so far:
                - Runs testing using `Test()` if a `testloader` is provided.
                - Computes evaluation metrics (MAE, MAPE, MSE, RMSE) on test data.
                - Logs the metrics.
                - Saves the model weights (`solution_u` and `dynamical_F`) as the best model.
                - Optionally saves the true and predicted labels to disk.
            7. Implements early stopping based on validation performance if `self.args.early_stop` is set.
            8. Clears the logger at the end of training to prevent duplicate logs in future runs.
            9. Saves the best model checkpoint to the specified folder if `self.args.save_folder` is provided.

        Args:
            trainloader (torch.utils.data.DataLoader): DataLoader providing training batches.
            testloader (torch.utils.data.DataLoader, optional): DataLoader for testing. Default is None.
            validloader (torch.utils.data.DataLoader, optional): DataLoader for validation. Default is None.

        Returns:
            None. 
            - Training, validation, and test metrics are logged via `self.logger`.
            - The best model is stored in `self.best_model` and optionally saved to disk.

        Notes:
            - Uses multiple loss components (data, PDE, physics) weighted by self.alpha and self.beta.
            - Early stopping prevents overfitting by stopping training if validation loss does not improve.
            - Learning rate scheduling is handled automatically for `solution_u` via `self.scheduler`.
        """
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        # Iterates through epochs
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            # Gets curernt losses
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            # Updates learning rate
            current_lr = self.scheduler.step()
            # Logs training info: epoch, lr, total loss
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            # Validates every epoch if validloader is provided
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            # Runs if current validation MSE is better than the best so far
            # Runs test set evaluation and saves the best model
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            # Early stopping check
            # Stops training if MSE has not improved for early_stop number of epochs
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        # Clears logger to avoid duplicates in future runs
        self.clear_logger()
        # Saves the best model to disk if a save folder is specified
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))




if __name__ == "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

        # scheduler 相关
        parser.add_argument('--epochs', type=int, default=1, help='epoch')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
        parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
        parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
        parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
        parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
        parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

        parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
        parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

        return parser.parse_args()


    args = get_args()
    pinn = PINN(args)
    print(pinn.solution_u)
    count_parameters(pinn.solution_u)
    print(pinn.dynamical_F)




