import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from torch.nn import Linear


class channelcl(nn.Module):
    def __init__(self, configs):
        super(channelcl, self).__init__()
        self.conv1 = nn.ModuleList(nn.Linear(configs.d_model, configs.d_model) for _ in range(configs.enc_in))
        self.conv2 = nn.ModuleList(nn.Linear(configs.d_model, configs.d_model) for _ in range(configs.enc_in))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(configs.dropout)
        self.norm = nn.LayerNorm(configs.d_model)
        self.channels = configs.enc_in

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device='cuda:0')
        for i in range(self.channels):
            o[:, i, :, :] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, i, :, :]))))
        res = o + x
        # res = self.norm(res)
        return res


class timecl(nn.Module):
    def __init__(self, configs, patnum):
        super(timecl, self).__init__()
        self.conv1 = nn.ModuleList(nn.Linear(patnum, patnum) for _ in range(configs.d_model))
        self.conv2 = nn.ModuleList(nn.Linear(patnum, patnum) for _ in range(configs.d_model))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(configs.dropout)
        self.norm = nn.LayerNorm(configs.d_model)
        self.channels = configs.d_model

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device='cuda:0')
        for i in range(self.channels):
            o[:, :, :, i] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, :, :, i]))))
        res = o + x
        # res = self.norm(res)
        return res


class patchnumcl(nn.Module):
    def __init__(self, configs, patnum):
        super(patchnumcl, self).__init__()
        self.conv1 = nn.ModuleList(nn.Linear(configs.d_model, configs.d_model) for _ in range(patnum))
        self.conv2 = nn.ModuleList(nn.Linear(configs.d_model, configs.d_model) for _ in range(patnum))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(configs.dropout)
        self.norm = nn.LayerNorm(configs.d_model)
        self.channels = patnum

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device='cuda:0')
        for i in range(self.channels):
            o[:, :, i, :] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, :, i, :]))))
        res = o + x
        # res = self.norm(res)
        return res


class Merging(nn.Module):
    def __init__(self):
        super(Merging, self).__init__()

    def merge(self, even_part, odd_part):
        '''Merge even and odd parts'''
        batch_size = even_part.size(0)
        even_len = even_part.size(2)
        hidden_size = even_part.size(1)
        odd_len = odd_part.size(2)

        # 初始化一个全零的张量，形状为 [batch_size, even_len+odd_len, hidden_size]
        merged = torch.zeros(batch_size, hidden_size, even_len + odd_len, dtype=even_part.dtype,
                             device=even_part.device)

        # 将 even_part 和 odd_part 分别填充回合并后的张量中的对应位置
        merged[:, :, ::2] = even_part
        merged[:, :, 1::2] = odd_part

        return merged

    def forward(self, even_part, odd_part):
        '''Merge even and odd parts back to original shape'''
        return self.merge(even_part, odd_part)


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, nf)
        self.linear4 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.gelu(self.linear1(x)) + x
        x = self.gelu(self.linear2(x)) + x
        x = self.gelu(self.linear3(x)) + x
        x = self.linear4(x)
        return x


class Model(nn.Module):  # 构建模型
    def __init__(self, config, **kwargs):
        super().__init__()
        self.model = TFPMixerformer(config)
        self.task_name = config.task_name

    def forward(self, x, *args, **kwargs):
        # 128,96,7  BS,L,C
        x = x.permute(0, 2, 1)
        # 128,7,96 BS,C,L
        mask = args[-1]
        x = self.model(x, mask=mask)
        if self.task_name != 'classification':
            x = x.permute(0, 2, 1)
        return x


class TFPMixerformer(nn.Module):
    def __init__(self,
                 config, **kwargs):

        super().__init__()
        self.patch_len = config.patch_len  # 16
        self.stride = config.stride  # 8
        self.d_model = config.d_model
        self.task_name = config.task_name
        self.channel = config.enc_in
        self.seq_len = config.seq_len

        self.patch_num = int((config.seq_len / 2 - self.patch_len) / self.stride + 1)
        self.patch_numf = int((config.seq_len - self.patch_len) / self.stride + 1)
        self.nnum = self.patch_num + 1

        self.W_pos_embed = nn.Parameter(torch.randn(self.patch_num, config.d_model) * 1e-2)
        self.W_pos_embedf = nn.Parameter(torch.randn(self.patch_numf, config.d_model) * 1e-2)
        self.model_token_number = 0
        # self.chanmixm = channelcl(config)
        # self.patchnummixm = patchnumcl(config, self.patch_numf+1)
        # self.timemixm = timecl(config, self.patch_numf+1)
        self.chanmixmf = channelcl(config)
        self.patchnummixmf = patchnumcl(config, self.patch_numf)
        self.timemixmf = timecl(config, self.patch_numf)
        self.chl = nn.Linear(self.channel, self.channel)
        self.dml = nn.Linear(self.d_model, self.d_model)
        self.pnl = nn.Linear(self.patch_numf, self.patch_numf)
        if self.model_token_number > 0:
            self.model_token = nn.Parameter(torch.randn(config.enc_in, self.model_token_number, config.d_model) * 1e-2)

        self.total_token_number = (self.patch_num + self.model_token_number + 1)
        config.total_token_number = self.total_token_number

        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        self.use_statistic = config.use_statistic
        self.W_statistic = nn.Linear(2, config.d_model)
        self.lineartoseq = nn.Linear(config.d_model * self.patch_numf, config.seq_len)
        self.seqdivide = nn.Linear(self.patch_len * 2, self.patch_len)
        self.breakn = nn.Linear(self.d_model, self.d_model)
        self.head = Flatten_Head(self.channel, self.d_model * self.patch_numf, self.seq_len, 0)
        self.dsdivide = nn.Linear(self.seq_len * 2, self.seq_len)
        self.cls = nn.Parameter(torch.randn(1, config.d_model) * 1e-2)

        if config.task_name == 'long_term_forecast' or config.task_name == 'short_term_forecast':
            self.W_out = nn.Linear(config.seq_len, config.pred_len)
        elif config.task_name == 'imputation' or config.task_name == 'anomaly_detection':
            self.W_out = nn.Linear(config.seq_len, config.seq_len)
        elif config.task_name == 'classification':
            self.W_out = nn.Linear(config.d_model * config.enc_in, config.num_class)

    def forward(self, z, *args, **kwargs):
        # 128 7 96

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'anomaly_detection':
            z_mean = torch.mean(z, dim=(-1), keepdims=True)
            z_std = torch.std(z, dim=(-1), keepdims=True)
            z = (z - z_mean) / (z_std + 1e-4)
        # 128 7 96 batchsize channel seqlength 先计算均值，再计算标准差，最后将数据-均值/标准差来标准化数据

        zz = torch.fft.fft(z)  # 正傅里叶变换，转换为频域，分为实部虚部
        z1 = zz.real  # 实部
        z2 = zz.imag  # 虚部
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 将实部，虚部分别进行patch操作
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        f = torch.cat((z1, z2), -1)  # 将实部，虚部合并
        f = self.seqdivide(f)  # 由于将最后一维放大了两倍，所以用一个线性层映射回去
        # batchsize channel patchnum patchlength



        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embedf
        f_embed = self.input_dropout(self.W_input_projection(
            f)) + self.W_pos_embedf  # w_input是进行将patchlen转换为dmodel然后进行dropout防止过拟合，然后通过位置嵌入矩阵增加信息

        #   cls_token = self.cls.repeat(z_embed.shape[0], z_embed.shape[1], 1, 1)
        # z_embed = torch.cat((cls_token, z_embed), dim=-2)  # 128, 7, 6, 64（c*(n+1)*d） 额外令牌矩阵+位置嵌入

        inputsf = f_embed
        inputs = z_embed
        b, c, t, h = inputs.shape

        inputs = self.timemixmf(inputs)
        outputs = self.chanmixmf(inputs)
        outputs = self.patchnummixmf(outputs)
        outputs = outputs.reshape(b, c, -1)
        outputs = self.lineartoseq(outputs)

        inputsf = self.timemixmf(inputsf)
        outputsf = self.chanmixmf(inputsf)
        outputsf = self.patchnummixmf(outputsf)

        f1 = self.breakn(outputsf)  # 这其实是一个同维度线性层映射
        f2 = self.breakn(outputsf)
        f1 = self.head(f1)  # 频率细化和归一化，沿信道进行非重叠的补片操作，防止模型过度关注具有较大振幅的分量，这个展平头还可以合并patch并将最后一位映射为想要的大小
        f2 = self.head(f2)  # 具体是建立残差连接，将经过线性层和relu激活函数的变量与原始数据相加，缓解梯度消失，梯度爆炸
        f = torch.fft.ifft(torch.complex(f1, f2))  # 反变换
        fr = f.real  # 分实部虚部进行合并，将维度再用线性层少映射一半
        fi = f.imag
        outputsf = self.dsdivide(torch.cat((fr, fi), -1))  # batchsize channel seqlen

        outputs = self.dsdivide(
            torch.cat((outputs, outputsf), -1))  # 将时域，频域信息整合，由于seqlen*2，用线性层缩小 batchsize channel seqlen
        if self.task_name != 'classification':
            z_out = self.W_out(outputs)  # seqlen转换为预测长度
            z = z_out * (z_std + 1e-4) + z_mean  # 反归一化
        else:
            outputs = outputs.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            z = self.W_out(torch.mean(outputs[:, :, :, :], dim=-2).reshape(b, -1))
        return z
