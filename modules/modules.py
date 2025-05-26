import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, layers_num, dropout=0.0, bias=True, base_activation=nn.ReLU):
        super().__init__()
        self.layers_num = layers_num
        self.dropout = False
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        h = [hidden_features] * (layers_num - 1)
        self.layers = nn.ModuleList(nn.Linear( i, o, bias=bias) for i, o in zip([in_features] + h, h + [out_features]))
        self.activation = base_activation()

    
    def forward(self, input):
        for i, layer in enumerate(self.layers):
            input = self.activation(layer(input)) if i < self.layers_num - 1 else layer(input)
            if self.dropout and i < self.layers_num - 1:
                input = self.dropout(input)
        return input


class Detector(nn.Module):
    expansion:int = 2
    def __init__(self, de_level, in_features, class_num, detection_num):
        super().__init__()
        self.detection_num = detection_num

        # self.left = nn.Linear(1, self.detection_num)

        self.class_predictor = MLP(in_features, self.expansion * in_features, class_num+1, 3)
        self.segment_predictor = MLP(in_features, self.expansion * in_features, 2, 3)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, input):
        bs, c, h, w = input.size()
        input = input.view(bs, 1, 1, c * h * w)

        # input = self.left(input.transpose(-1, -2))
        # input = input.transpose(-1, -2)
        input = input.repeat(1, 1, self.detection_num, 1)

        class_set = self.class_predictor(input)
        segment_set = self.segment_predictor(input)

        segment_set = self.sigmoid(segment_set)

        class_set = class_set.squeeze(1)
        segment_set = segment_set.clone().squeeze(1)
        return {'class_set': class_set, 'segment_set': segment_set}


class Forecastor(nn.Module):
    def __init__(self, in_features, forecast_length):
        super().__init__()
        self.forcastor = nn.Linear(in_features, forecast_length)



class Classifier(nn.Module):
    def __init__(self, de_level, in_features, class_num, detection_num):
        super().__init__()
        self.class_predictor_left_linear = nn.Linear(de_level, detection_num)
        self.class_predictor = nn.Linear(in_features, class_num)

    def forward(self, input):
        class_out = self.class_predictor_left_linear(input.transpose(2,3)).transpose(2,3)
        class_set = F.softmax(self.class_predictor(class_out).transpose(2,3), dim=-1)
        return class_set



