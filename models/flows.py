import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

#code from https://github.com/nkolot/nflows/blob/master/nflows
class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, affine=True) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps

class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=None,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)
        stdv = 0.01 / np.sqrt(hidden_features)
        init.uniform_(self.final_layer.weight, -stdv, stdv)
        init.uniform_(self.final_layer.bias, -stdv, stdv)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=None)
        outputs = self.final_layer(temps)
        return outputs

class CondResAdditiveCouplingLayer(nn.Module):
    
    def __init__(self,cond_feat_dim,feat_dim):
        super().__init__()

        self.feat_dim = feat_dim
        self.net = ResidualNet(feat_dim//2,feat_dim//2,1024,cond_feat_dim)
    
    def forward(self,x,cond,reverse=False):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Found nan/inf in x")
            import pdb;pdb.set_trace()
        x1,x2 = torch.split(x,self.feat_dim//2,dim=-1)
        y1=x1
        t = self.net(x1,cond)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print("Found nan/inf in t")
            import pdb;pdb.set_trace()

        log_det = 0.0

        if not reverse:
            y2 = x2 + t
            if torch.isnan(y2).any() or torch.isinf(y2).any():
                print("Found nan/inf in y2")
                import pdb;pdb.set_trace()
        else:
            y2 = x2-t

        y = torch.concat([y1,y2],dim=-1)
        return y,log_det

#code adapted from https://github.com/nkolot/nflows/blob/master/nflows/transforms/lu.py
class LULinear(nn.Module):
    def __init__(self,feat_dim):
        super().__init__()

        self.eps = 1e-3
        self.feat_dim = feat_dim

        self.lower_ids = np.tril_indices(feat_dim,k=-1)
        self.upper_ids = np.triu_indices(feat_dim,k=1)
        self.diag_ids = np.diag_indices(feat_dim)

        num_params = (feat_dim*(feat_dim-1))//2

        self.lower_entries = torch.nn.Parameter(torch.zeros(num_params))
        self.upper_entries = torch.nn.Parameter(torch.zeros(num_params))
        self.unconstrained_diag = torch.nn.Parameter(torch.zeros(feat_dim))

        self._initialize()

    def _initialize(self):

        init.zeros_(self.lower_entries)
        init.zeros_(self.upper_entries)
        constant = np.log(np.exp(1-self.eps)-1)
        init.constant_(self.unconstrained_diag,constant)

    def create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.feat_dim,self.feat_dim)
        lower[self.lower_ids[0],self.lower_ids[1]] = self.lower_entries
        lower[self.diag_ids[0],self.diag_ids[1]] = 1.0

        upper = self.upper_entries.new_zeros(self.feat_dim,self.feat_dim)
        upper[self.upper_ids[0],self.upper_ids[1]] = self.upper_entries
        upper[self.diag_ids[0],self.diag_ids[1]] = self.upper_diag
        
        return lower,upper

    def weight(self):
        l,u = self.create_lower_upper()
        return l @ u

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_diag) + self.eps

    def logabsdet(self):
        return torch.sum(torch.log(self.upper_diag))

    def forward(self,x,reverse=False):

        lower,upper = self.create_lower_upper()
        log_det = self.logabsdet() * x.new_ones(x.shape[0])
        
        if not reverse:
            y = F.linear(x,lower)
            y = F.linear(y,upper)

        else:
            y,_ = torch.triangular_solve(x.t(),lower,upper=False,unitriangular=True)
            y,_ = torch.triangular_solve(y,upper,upper=True,unitriangular=False)
            y = y.t()

            log_det = - log_det
    
        return y,log_det

class ActNorm(nn.Module):

    def __init__(self,feat_dim):
        super().__init__()

        self.logs = nn.Parameter(torch.randn(feat_dim)) 
        self.bias = nn.Parameter(torch.randn(feat_dim))
        self.register_buffer("initialized",torch.tensor(False))

    def initilization(self,batch):
        print("LOG:: Initializing Act Norm Layer")
        self.initialized = torch.tensor(True)

        with torch.no_grad():
            print("LOG:: Initializing from {0} samples".format(batch.shape[0]))
            mean = batch.mean(dim=0)

            if batch.shape[0] == 1:
                std = torch.zeros_like(batch,device=batch.device)
            else:
                std = batch.std(dim=0)

            #std = torch.maximum(std,1e-5*torch.ones_like(std,device=std.device))

            bias = -mean/(std)
            logs = - torch.log(std) 
            self.bias.data.copy_(bias.squeeze(0))
            self.logs.data.copy_(logs.squeeze(0))

    def forward(self,x,reverse=False):
        if not self.initialized and self.training:
            self.initilization(x)
            
        if not reverse:
            y = x*torch.exp(self.logs) + self.bias
            log_det = torch.sum(self.logs,dim=-1)
        else:
            y = (x-self.bias)/torch.exp(self.logs)
            log_det = -torch.sum(self.logs,dim=-1)

        if torch.isnan(log_det).any() or torch.isinf(log_det).any():
            print("Found nan/inf in log_det of actnorm")
            import pdb;pdb.set_trace()


        return y,log_det

class CondFlowBlock(nn.Module):
    
    def __init__(self,feat_dim,cond_dim):
        super().__init__()

        self.adflow = CondResAdditiveCouplingLayer(cond_dim,feat_dim)
        self.conv = LULinear(feat_dim)
        self.actnorm = ActNorm(feat_dim)

    def forward(self,x,cond,reverse=False):
        if not reverse:
            x,ld1 = self.actnorm(x)
            x,ld2 = self.conv(x)
            x,ld3 = self.adflow(x,cond)

        else:
            x,ld3 = self.adflow(x,cond,reverse=True)
            x,ld2 = self.conv(x,reverse=True)
            x,ld1 = self.actnorm(x,reverse=True)
        
        return x, ld1+ld2+ld3