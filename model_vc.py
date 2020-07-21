import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bias=True, nonlinearity='linear'):
        # bias should be False
        assert (kernel_size % 2 == 1)
        padding = int((kernel_size - 1) / 2)

        if nonlinearity not in ('linear', 'relu', 'tanh'):
            raise ValueError('nonlinearity should be linear, relu or tanh')

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels),
        ]

        if nonlinearity == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif nonlinearity == 'tanh':
            layers += [nn.Tanh()]

        super(ConvBlock, self).__init__(*layers)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlinearity))


class Encoder(nn.Module):
    """Encoder module:
    """

    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        layers = [ConvBlock(80 + dim_emb, 512, kernel_size=5, nonlinearity='relu')]
        for _ in range(2):
            layers += [ConvBlock(512, 512, kernel_size=5, nonlinearity='relu')]
        self.conv = nn.Sequential(*layers)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2, 1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)

        x = self.conv(x)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        return codes


class Decoder(nn.Module):
    """Decoder module"""

    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True)

        layers = []
        for _ in range(3):
            layers += [ConvBlock(dim_pre, dim_pre, kernel_size=5, nonlinearity='relu')]
        self.conv = nn.Sequential(*layers)

        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)

        self.linear_projection = nn.Linear(1024, 80)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)

        decoder_output = self.linear_projection(outputs)

        return decoder_output


class Postnet(nn.Sequential):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        layers = [ConvBlock(80, 512, kernel_size=5, nonlinearity='tanh')]
        for _ in range(3):
            layers += [ConvBlock(512, 512, kernel_size=5, nonlinearity='tanh')]
        layers += [ConvBlock(512, 80, kernel_size=5, nonlinearity='linear')]
        super(Postnet, self).__init__(*layers)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)

        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)

        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)

        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)
