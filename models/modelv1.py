import torch
import torch.nn as nn
import torch.nn.functional as f
import fracdiff.torch.functional as fdf
from modules.modules import MLP
from models import ModelBase
import swanlab

class MODS(nn.Module):
    def __init__(self, d,  length, window=10, resolution=0.1, eps=1e-6):
        super().__init__()
        self.d = d
        self.length = length
        self.window = window
        self.resolution = resolution
        self.eps = eps
        self.orders = torch.arange(0, self.d + self.eps, self.resolution) + self.eps
        self.net = nn.Sequential(
            nn.LayerNorm([self.length, self.orders.shape[-1]], self.eps),
            nn.Linear(self.orders.shape[-1], self.orders.shape[-1] * 2),
            nn.GELU(),
            nn.Linear(self.orders.shape[-1] * 2, self.orders.shape[-1]),
        )

    def forward(self, x):
        diffs = torch.stack([fdf.fdiff(x, n.item(), -1, window=self.window, mode="same") for n in self.orders], -1)
        out =  x + self.net(diffs).mean(-1)
        return out



class SFSD(nn.Module):
    def __init__(self, deepth, topk, trend, length, eps=1e-6):
        super().__init__()
        self.deepth = deepth
        self.topk = topk
        self.trend = trend
        self.eps = eps
        self.ln = nn.LayerNorm(length)
    
    
    def __decomp(self, input, t):
        # input: [B, N, L]
        B, N, L = input.shape
        if L % 2:
            x = torch.nn.functional.pad(input, (0, 1, 0, 0))
        else:
            x = input
        rfft_x = torch.fft.rfft(x, dim=-1)
        raw_fft_x = rfft_x.clone()
        rfft_x[:, :, :t] = 0
        period_rfft_x = rfft_x.clone()
        psd_x = torch.square(torch.abs(rfft_x))
        _, __, l = psd_x.shape
        power, freq = torch.topk(psd_x, l)
        power_per = torch.cumsum(power, dim=-1) / torch.sum(power, dim=-1, keepdim=True)
        freq[power_per > self.topk] = 0
        period_rfft_x[freq == 0] = 0
        rfft_res = rfft_x - period_rfft_x
        rfft_trend = raw_fft_x - period_rfft_x - rfft_res
        period_x = torch.fft.irfft(period_rfft_x, n=x.shape[-1])
        res = torch.fft.irfft(rfft_res, n=x.shape[-1])
        trend = torch.fft.irfft(rfft_trend, n=x.shape[-1])
        if L % 2:
            trend = trend[:, :, :-1]
            period_x = period_x[:, :, :-1]
            res = res[:, :, :-1]
        return trend, period_x, res
    
    def norm(self, input):
        std, m = torch.std_mean(input, dim=-1, keepdim=True)
        out = (input - m) / (std + self.eps)
        return out, m, std
    
    @torch.no_grad()
    def forward(self, x, norm=True):
        mod_list = []
        res_total = torch.zeros_like(x, device=x.device)
        for i in range(1, self.deepth + 1):
            trend, x, res = self.__decomp(x, self.trend * i)
            mod_list.append(trend)
            res_total += res
        mod_list.append(x)
        mod_list.append(res_total)
        out = torch.stack(mod_list, dim=2).contiguous()
        if norm:
            out = self.ln(out)
        return out
    

class ResFFN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.net = MLP(in_features, hidden_features, out_features, layers_num=2, base_activation=nn.GELU)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        b, c, m, l = x.shape
        if m == self.in_features:
            _shape = (0, 1, 3, 2)
        elif c == self.in_features:
            _shape = (0, 3, 2, 1)
        elif l == self.in_features:
            _shape = (0, 1, 2, 3)
        else:
            raise ValueError(x.shape, self.in_features)
        out =  self.net(x.permute(_shape)).permute(_shape)   
        return x + self.dropout(out)  

class ChannelMixer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.in_features = in_features
        self.net = MLP(in_features, hidden_features, out_features, layers_num=2, base_activation=nn.GELU)
    
    def forward(self, x):
        return self.net(x.transpose(-2, -1)).transpose(-2, -1)



class NLMDFE(nn.Module):
    expansion = 2
    def __init__(self, length, num_mods, num_channel, dropout):
        super().__init__()
        self.net = nn.Sequential(
            ResFFN(length, length, self.expansion * length, dropout),
            ResFFN(num_mods, num_mods, self.expansion * num_mods, dropout),
        )
        self.TM = ResFFN(length, length, self.expansion * length, dropout)
        self.MM = ResFFN(num_mods, num_mods, self.expansion * num_mods, dropout)
        self.CM = ChannelMixer(num_channel, num_channel, self.expansion * num_channel)


    def forward(self, x):
        tm_out = self.TM(x)
        mm_out = self.MM(tm_out).mean(-2)
        cm_out = self.CM(mm_out)
        return mm_out + f.relu(cm_out)
    
class _DecompNet4SEDV1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = nn.Sequential(
            SFSD(args.diff_order, args.k, args.t, args.seq_len),
            MODS(args.d, args.seq_len, args.window, args.resolution),
            NLMDFE(args.seq_len, args.diff_order + 2, args.num_features, args.dropout),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm((args.num_features, args.seq_len)),
            nn.Flatten(),
            MLP(args.num_features * args.seq_len, args.seq_len, args.num_classes, 2),
        )
        self.apply_headers = [None, None, self.classifier, None, None]

    def forward(self, x):
        feats = self.net(x)
        outputs = self.apply_headers[self.args.task](feats)
        return {
            "outputs": outputs,
            "feats": feats,
        }
    

    
class DecompNet4SEDV1(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.model = _DecompNet4SEDV1(args)


    def forward(self, x):
        return self.model(x)
    
    def kernel(self, samples, targets):
        samples = samples.to(self.device)
        targets = targets.to(self.device)
        outputs_dict = self.forward(samples)
        loss = self.criterion(outputs_dict['outputs'], targets)
        outputs_dict.update({"loss": loss, "targets": targets})
        return outputs_dict