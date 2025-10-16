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

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
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
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
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
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
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




