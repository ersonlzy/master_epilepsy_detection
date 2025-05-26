import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.modules import MLP
from models import ModelBase

class ResMixerBlock(nn.Module):
    def __init__(self, num_features, length, diff_order, d_model, pre_len, d, dropout, channel_independent=True, base_activation=nn.GELU):
        self.expansion = 4
        super().__init__()
        self.diffMixer = nn.Sequential(
            nn.Linear(diff_order, self.expansion * diff_order),
            base_activation(),
            nn.Linear(self.expansion * diff_order, diff_order),
            nn.Dropout(dropout),
        )
        
        self.tempMixer = nn.Sequential(
            # nn.BatchNorm2d(num_features),
            nn.Linear(length, d_model), 
            base_activation(),
            nn.Linear(d_model, length),
            nn.Dropout(dropout)
        )
        
        self.modMixer = nn.Sequential(
            # nn.BatchNorm2d(num_features),
            nn.Linear(d * 2 + 1, self.expansion * (d * 2 + 1)), 
            base_activation(),
            nn.Linear(self.expansion * (d * 2 + 1), d * 2 + 1),
            nn.Dropout(dropout)
        )
        
        if channel_independent:
            self.ChannelMixer = None
        else:
            # self.channelNorm = nn.BatchNorm2d(num_features)
            self.ChannelMixer = nn.Sequential(
                nn.Linear(num_features, num_features * self.expansion), 
                base_activation(),
                nn.Linear(num_features * self.expansion, num_features),
                nn.Dropout(dropout)
            )
            
        
    def forward(self ,input):
        # B, N, D, O, L
        b, n, d, o, l  = input.shape 
        out = input.transpose(-2, -1)
        out = input + self.diffMixer(out).transpose(-2, -1)
        out = input.sum(dim=3) / o
        out = out + self.tempMixer(out)
        out = out + self.modMixer(out.transpose(-2, -1)).transpose(-2, -1)
        if self.ChannelMixer:
            out = torch.permute(self.ChannelMixer(torch.permute(out, (0, 2, 3, 1))), (0, 3, 1, 2))
        return out  # B, N, D, L
        
        
        
class MS(nn.Module):
    def __init__(self, channels, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((channels, in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((channels, out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        # input.shape: B, N, D, L
        # print(input.shape, self.weight.shape)
        out = torch.einsum('...nij, njk -> ...nik', input, self.weight)
        # out = torch.einsum('...mij, mjk -> ...mik', input, self.weight)
        if self.bias is not None:
            out = out + self.bias
        # print(out.shape)
        return out
        

class MLinear(nn.Module):
    def __init__(self, channels, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((channels, in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((channels, out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        # input.shape: B, N, D, L
        out = torch.einsum('...nmj, mjk -> ...nmk', input, self.weight)
        # out = torch.einsum('...mij, mjk -> ...mik', input, self.weight)
        if self.bias is not None:
            out = out + self.bias
        # print(out.shape)
        return out
            
class EmptyAct(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input

class MMLP(nn.Module):
    def __init__(self, channels, in_features, out_features, hidden_features, layers_num, dropout=0.0, bias=True, base_activation=nn.GELU):
        super().__init__()
        self.layers_num = layers_num
        self.dropout = False
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        h = [hidden_features] * (layers_num - 1)
        self.layers = nn.ModuleList(MLinear( channels, i, o, bias=bias) for i, o in zip([in_features] + h, h + [out_features]))
        self.activation = base_activation()
        
        
    def forward(self, input):
        for i, layer in enumerate(self.layers):
            input = self.activation(layer(input)) if i < self.layers_num - 2 else layer(input)
            if self.dropout:
                input = self.dropout(input)
            
        return input
    
    
    
class SeriesDecomp(nn.Module):
    def __init__(self, k:int):
        super().__init__()
        self.k = k
        
    def forward(self, input):
        # input shape: B, diff_order, L
        rfft_input = torch.fft.rfft(input)
        abs_rfft_input = torch.abs(rfft_input)
        v_k, _ = torch.topk(abs_rfft_input, k=self.k)
        v_min, _ = torch.min(v_k, dim=-1)
        # rfft_season = deepcopy(rfft_input)
        rfft_input[abs_rfft_input <= v_min.unsqueeze(dim=-1)] = 0
        return torch.fft.irfft(rfft_input)
        
    
class Model(nn.Module):
    expansion = 4
    def __init__(self, args):
        super().__init__()
        self.args = args
        pred_len = args.pred_len
        num_features = args.num_features
        k = args.k
        d_model = args.d_model
        t = args.t
        d = args.d
        task = args.task
        length = args.seq_len
        dropout = args.dropout
        channel_independent = args.channel_independent
        num_classes = args.num_classes
        diff_order = args.diff_order
        
        self.pre_len = pred_len
        self.diff_order = diff_order
        self.d_model = d_model
        self.k = k
        self.t = t
        self.d = d
        self.task = task
        
        
        if task == 0 or task == 1:
            self.Mixer = ResMixerBlock(num_features, length, diff_order+1, d_model, pred_len, d, dropout=dropout, channel_independent=channel_independent, base_activation=nn.GELU)
            # self.ModPredictor = MLinear(self.d * 2 + 1, length, pred_len)
            self.ModPredictor = MMLP(self.d * 2 + 1, length, pred_len, d_model, 1, base_activation=nn.GELU)
        elif task == 2:
            self.Mixer = ResMixerBlock(num_features, length, diff_order+1, d_model, pred_len, d, dropout=dropout, channel_independent=channel_independent, base_activation=nn.GELU)
            self.ln = nn.LayerNorm((self.d * 2 + 1, length))
            self.classifyHeader = MLP((self.d * 2 + 1) * length, length, num_classes, 2)
            # self.classifyHeader = MMLP(num_features, (self.d * 2 + 1) * length, num_class, d_model, 1, base_activation=nn.GELU)
        elif task == 3:
            self.Mixer = ResMixerBlock(num_features, length, diff_order+1, d_model, pred_len, d, dropout=dropout, channel_independent=channel_independent, base_activation=nn.GELU)
            self.abnormalDetectHeader = MLP((self.d * 2 + 1) * length, (self.d * 2 + 1) * length, length, 1)
        elif task == 4:
            self.Mixer = ResMixerBlock(num_features, length, diff_order+1, d_model, pred_len, d, dropout=dropout, channel_independent=channel_independent, base_activation=nn.GELU)
            self.stsfHeader = MLP((self.d * 2 + 1) * length, d_model, pred_len, 2)
        
        self.reset_params()    

        
        
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    
    def forecast(self, input):
        out, m, std = self.__decompEmbedding(input)  # B, N, D, 
        # out = out * std + m
        out = self.__diffEmbedding(out)      # B, N, D, L, O
        feats = self.Mixer(out)
        # feats = self.ln(feats)
        # feats = feats * std + m
        out = self.ModPredictor(feats)
        out = out * std + m
        out = torch.sum(out, dim=2)         # B, N, D, L

        outputs_dict = {
            'moving mean': out,
            'moving var': out,
            'season': out,
            'random walk': out,
            'res': out,
            'target': out,
        }
        inputs_dict = {
            'moving mean': input,
            'moving var': input,
            'season': input,
            'random walk': input,
            'res': input,
            'target': input,
        }
        return out, outputs_dict, inputs_dict
    
    def stsf(self, input):
        out = self.__decompEmbedding(input)  # B, N, D, L
        out = self.__diffEmbedding(out)      # B, N, D, L, O
        feats = self.Mixer(out)
        out = self.stsfHeader(torch.flatten(feats, 1))

        outputs_dict = {
            'moving mean': out,
            'moving var': out,
            'season': out,
            'random walk': out,
            'res': out,
            'target': out,
        }
        inputs_dict = {
            'moving mean': input,
            'moving var': input,
            'season': input,
            'random walk': input,
            'res': input,
            'target': input,
        }
        return out, outputs_dict, inputs_dict
    
    
    def imputation(self, input, mask):
        out = self.__decompEmbedding(input)  # B, N, D, L
        out = self.__diffEmbedding(out)      # B, N, D, L, O
        feats = self.Mixer(out)
        out = feats.sum(dim=-2)
        # out = self.pro(feats.sum(dim=-2))
        
        outputs_dict = {
            'moving mean': out,
            'moving var': out,
            'season': out,
            'random walk': out,
            'res': out,
            'target': out,
        }
        inputs_dict = {
            'moving mean': input,
            'moving var': input,
            'season': input,
            'random walk': input,
            'res': input,
            'target': input,
        }
        return out, outputs_dict, inputs_dict
    
    
    def classify(self, input):
        out, m, std = self.__decompEmbedding(input)  # B, N, D, L
        out = out * std + m
        out = self.__diffEmbedding(out)      # B, N, D, O, L
        feats = self.Mixer(out)
        # print(feats.shape)
        feats = self.ln(feats)
        feats = torch.sum(feats, dim=1)
        out = self.classifyHeader(torch.flatten(feats, 1))
        # print(out.shape)
        # out = torch.sum(out, dim=1)
        return out
    
    
    def abnormalDetect(self, input):
        out, m, std = self.__decompEmbedding(input)  # B, N, D, L
        out = out * std + m
        out = self.__diffEmbedding(out)      # B, N, D, L, O
        feats = self.Mixer(out)
        out = self.abnormalDetectHeader(torch.flatten(feats, 2))
        return out
        
        
    def decomp(self, input):
        return input, input, input, input, input
    
    def forward(self, x, mask=None):
        if self.task == 0:
            return self.forecast(x)
        elif self.task == 1:
            return self.imputation(x, mask)
        elif self.task == 2:
            return self.classify(x)
        elif self.task == 3:
            return self.abnormalDetect(x)
        elif self.task == 4:
            return self.stsf(x)
        else:
            raise NotImplementedError
        
    @torch.no_grad()
    def __decomp(self, input, t):
        # x.shape: B, N, L
        B, N, L = input.shape
        if L % 2:
            x = torch.nn.functional.pad(input, (0,1,0,0))
        else:
            x = input
        eps = 1
        rfft_x = torch.fft.rfft(x, dim=-1)
        raw_fft_x = torch.clone(rfft_x)
        rfft_x[:, :, :t] = 0
        period_rfft_x = torch.clone(rfft_x)
        psd_x = torch.square(torch.abs(rfft_x))
        _, __, l = psd_x.shape
        power, freq = torch.topk(psd_x, l)
        power_per = torch.cumsum(power, dim=-1) / torch.sum(power, dim=-1, keepdim=True)
        freq[power_per > self.k] = 0
        period_rfft_x[freq==0] = 0
        rfft_res = rfft_x - period_rfft_x
        rfft_trend = raw_fft_x - period_rfft_x - rfft_res
        period_x = torch.fft.irfft(period_rfft_x)
        res = torch.fft.irfft(rfft_res)
        trend = torch.fft.irfft(rfft_trend)
        if L % 2:
            trend = trend[:,:,:-1]
            period_x = period_x[:,:,:-1]
            res = res[:,:,:-1]
        
        return trend, period_x, res
    

    
    @torch.no_grad()
    def __decompEmbedding(self, x):
        mod_list = []
        for i in range(1, self.d + 1):
            trend, x, res = self.__decomp(x, self.t * i)
            mod_list.append(trend)
            mod_list.append(res)
        mod_list.append(x)
        out = torch.stack(mod_list, dim=2).contiguous()
        out, m, std = self.norm(out)
        return out, m, std
        
        
    def __phaseEncoding(self, x):
        phase = torch.angle(self.__hilbert(x))/ torch.pi
        return x + phase
    
    @torch.no_grad()
    def __diffEmbedding(self, x:torch.Tensor):
        out = torch.stack([self.__norm(F.pad(torch.diff(x, n=i, dim=-1), pad=(0, i), mode='constant', value=0)) for i in range(0, self.diff_order+1)], dim=3).contiguous()
        out = x.unsqueeze(dim=3) + out
        return out
    
    def __maxMinScale(self, input:torch.Tensor):
        eps = 1e-6
        max, _ = input.max(dim=-1, keepdim=True)
        min, _ = input.min(dim=-1, keepdim=True)
        out = (input - min) / (max - min + eps)
        return out
    
    def __norm(self, input):
        eps = 1e-6
        std, m = torch.std_mean(input, keepdim=True)
        out = (input - m) / (std + eps)
        return out
    
    def norm(self, input):
        eps = 1e-6
        std, m = torch.std_mean(input, dim=-1, keepdim=True)
        out = (input - m) / (std + eps)
        return out, m, std
    
    def __no(self, input):
        return input
    
    
    def __hilbert(self, x):
        b, n, d, l = x.shape
        fft_x = torch.fft.fft(x, dim=-1)
        h = torch.zeros(fft_x.shape, dtype=fft_x.dtype, device=x.device)
        if l % 2 == 0:
                h[:, :, :, 0] = h[:, :, :, l // 2] = 1
                h[:, :, : , 1:l // 2] = 2
        else:
                h[:, :, :, 0] = 1
                h[:, :, :, 1:(l + 1) // 2] = 2
                
        hil_x = torch.fft.ifft(fft_x * h, dim=-1)
        return hil_x
    



class DecompNet4ESD(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.model = Model(args)

    def forward(self, x):
        return self.model(x)