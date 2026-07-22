import torch
import torch.nn as nn
import torch.nn.functional as F
model_urls = {
    'scnet50_v1d': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50_v1d-4109d1e1.pth',
}

def hippo_dplr_init(state_dim, max_N=64):
    N = min(state_dim, max_N)
    arange = torch.arange(1, N + 1, dtype=torch.float32)

    real_part = -0.5 * (2 * arange - 1)
    imag_part = torch.pi * (arange - 0.5)
    eigenvalues = torch.complex(real_part, imag_part)

    Lambda = torch.diag(eigenvalues)

    p = torch.randn(N, 1, dtype=torch.complex64)
    q = torch.randn(N, 1, dtype=torch.complex64)
    A = Lambda + p @ q.T

    if state_dim > N:
        pad = state_dim - N
        A = F.pad(A, (0, pad, 0, pad), value=0)

    return A, torch.tensor(1.0)

class SSM_HiPPO(nn.Module):
    def __init__(
            self,
            state_dim=64,
            input_dim=None,
            discretization="zoh",
            dt_init=0.001,
            init_std=0.012,
            n_segment=8,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.discretization = discretization
        self.n_segment = n_segment
        self.input_dim = input_dim
        A_cont, _ = hippo_dplr_init(state_dim)
        self.register_buffer("A_cont", A_cont)
        self.dt = nn.Parameter(torch.full((1,), dt_init))

        if input_dim is not None:
            self.B = nn.Linear(input_dim, state_dim, bias=False)
            self.C = nn.Linear(state_dim, input_dim, bias=False)
            self._init_BC(init_std)
        else:
            self.B = nn.Linear(1, state_dim, bias=False)
            self.C = nn.Linear(state_dim, 1, bias=False)
            self._input_dim_pending = True

        self.D = nn.Parameter(torch.zeros(1))

        self.init_state = nn.Parameter(torch.zeros(1, state_dim))
        self.norm_layer = norm_layer

    def _init_BC(self, std):
        nn.init.normal_(self.B.weight, 0, std)
        nn.init.normal_(self.C.weight, 0, std)

    def _discretize(self, A, B, dt):
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

        if self.discretization == "zoh":
            A_d = torch.matrix_exp(dt * A)
            B_d = torch.linalg.solve(A, (A_d - I) @ B)
        elif self.discretization == "bilinear":
            A_d = torch.linalg.solve(I - dt / 2 * A, I + dt / 2 * A)
            B_d = torch.linalg.solve(I - dt / 2 * A, dt * B)
        else:
            raise ValueError

        return A_d.real, B_d.real

    def forward(self, x):
        nt, c, h, w = x.shape
        t = nt // (nt // self.n_segment)
        n = nt // t
        input_dim = c * h * w

        if hasattr(self, "_input_dim_pending"):
            assert self.input_dim is None or self.input_dim == input_dim, \
                "Input resolution mismatch after first forward"
            self.input_dim = input_dim
            self.B = nn.Linear(input_dim, self.state_dim, bias=False).to(x.device)
            self.C = nn.Linear(self.state_dim, input_dim, bias=False).to(x.device)
            self._init_BC(0.012)
            del self._input_dim_pending

        u = x.view(n, t, input_dim)
        B_cont = self.B.weight.to(torch.complex64)
        A_d, B_d = self._discretize(self.A_cont, B_cont, self.dt.abs())
        state = self.init_state.expand(n, -1)
        outputs = []

        for i in range(t):
            state = state @ A_d.T + u[:, i] @ B_d.T
            y = self.C(state).view(n, c, h, w)
            outputs.append(y)

        out = torch.cat(outputs, dim=1).view(nt, c, h, w)
        out = out + self.D * x
        return out, out

    def extra_repr(self):
        return (
            f"state_dim={self.state_dim}, "
            f"input_dim={self.input_dim}, "
            f"discretization={self.discretization}, "
            f"dt={self.dt.item():.4f}"
        )

class ISM_Module(nn.Module):
    def __init__(self, k_size=3):

        super(ISM_Module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class TIM_Module(nn.Module):
    def __init__(self, in_channels, reduction=16, n_segment=8, return_attn=False):
        super(TIM_Module, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.n_segment = n_segment
        self.return_attn = return_attn

        self.reduced_channels = self.in_channels // self.reduction

        self.conv1 = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.reduced_channels)

        self.conv_ht = nn.Conv2d(self.reduced_channels, self.reduced_channels,
                                 kernel_size=(3, 1), padding=(1, 0), groups=self.reduced_channels, bias=False)
        self.conv_tw = nn.Conv2d(self.reduced_channels, self.reduced_channels,
                                 kernel_size=(1, 3), padding=(0, 1), groups=self.reduced_channels, bias=False)

        self.avg_pool_ht = nn.AvgPool2d((2, 1), (2, 1))
        self.avg_pool_tw = nn.AvgPool2d((1, 2), (1, 2))

        self.htie_conv1 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.vtie_conv1 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.htie_conv2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.vtie_conv2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.ht_up_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )
        self.tw_up_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def feat_ht(self, feat):
        n, t, c, h, w = feat.size()
        feat_h = feat.permute(0, 4, 2, 3, 1).contiguous().view(-1, c, h, t)

        feat_h_fwd, _ = feat_h.split([self.n_segment - 1, 1], dim=3)
        feat_h_conv = self.conv_ht(feat_h)
        _, feat_h_conv_fwd = feat_h_conv.split([1, self.n_segment - 1], dim=3)

        diff_feat_fwd = feat_h_conv_fwd - feat_h_fwd
        diff_feat_fwd = F.pad(diff_feat_fwd, [0, 1], value=0)

        diff_feat_fwd1 = self.avg_pool_ht(diff_feat_fwd)
        diff_feat_fwd1 = self.htie_conv1(diff_feat_fwd1)
        diff_feat_fwd1 = F.interpolate(diff_feat_fwd1, diff_feat_fwd.size()[2:])
        diff_feat_fwd2 = self.htie_conv2(diff_feat_fwd)

        feat_ht_out = self.ht_up_conv(1 / 3. * diff_feat_fwd + 1 / 3. * diff_feat_fwd1 + 1 / 3. * diff_feat_fwd2)
        feat_ht_out = self.sigmoid(feat_ht_out) - 0.5
        feat_ht_out = feat_ht_out.view(n, w, self.in_channels, h, t).permute(0, 4, 2, 3, 1).contiguous()
        feat_ht_out = feat_ht_out.view(-1, self.in_channels, h, w)

        return feat_ht_out

    def feat_tw(self, feat):
        n, t, c, h, w = feat.size()
        feat_w = feat.permute(0, 3, 2, 1, 4).contiguous().view(-1, c, t, w)

        feat_w_fwd, _ = feat_w.split([self.n_segment - 1, 1], dim=2)
        feat_w_conv = self.conv_tw(feat_w)
        _, feat_w_conv_fwd = feat_w_conv.split([1, self.n_segment - 1], dim=2)

        diff_feat_fwd = feat_w_conv_fwd - feat_w_fwd
        diff_feat_fwd = F.pad(diff_feat_fwd, [0, 0, 0, 1], value=0)

        diff_feat_fwd1 = self.avg_pool_tw(diff_feat_fwd)
        diff_feat_fwd1 = self.vtie_conv1(diff_feat_fwd1)
        diff_feat_fwd1 = F.interpolate(diff_feat_fwd1, diff_feat_fwd.size()[2:])
        diff_feat_fwd2 = self.vtie_conv2(diff_feat_fwd)

        feat_tw_out = self.tw_up_conv(1 / 3. * diff_feat_fwd + 1 / 3. * diff_feat_fwd1 + 1 / 3. * diff_feat_fwd2)
        feat_tw_out = self.sigmoid(feat_tw_out) - 0.5
        feat_tw_out = feat_tw_out.view(n, h, self.in_channels, t, w).permute(0, 3, 2, 1, 4).contiguous()
        feat_tw_out = feat_tw_out.view(-1, self.in_channels, h, w)

        return feat_tw_out

    def forward(self, x):
        bottleneck = self.conv1(x)
        bottleneck = self.bn1(bottleneck)
        bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])
        F_h = self.feat_ht(bottleneck)
        F_w = self.feat_tw(bottleneck)
        att = 0.5 * (F_h + F_w)
        if self.return_attn:
            return att
        y2 = x + x * att
        return y2

class ShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div * self.fold, self.fold_div * self.fold,
                              kernel_size=3, padding=1, groups=self.fold_div * self.fold,
                              bias=False)
        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1
            if 2 * self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1)
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)
        x = self.conv(x)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2)
        x = x.contiguous().view(nt, c, h, w)
        return x

class TSBM(nn.Module):
    expansion = 4
    pooling_r = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None):
        super(TSBM, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)
        self.tim = TIM_Module(group_width, n_segment=num_segments)
        self.shift = ShiftModule(group_width, n_segment=num_segments, n_div=8, mode='shift')
        self.inplanes = inplanes
        self.planes = planes
        self.ism = ISM_Module()
        self.shift = ShiftModule(group_width, n_segment=num_segments, n_div=8, mode='shift')

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False),
            norm_layer(group_width),
        )
        self.ssm = SSM_HiPPO(
            state_dim=64, input_dim=None, n_segment=num_segments, dt_init=0.001, init_std=0.012, )
        self.conv3 = nn.Conv2d(
            group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out_a = self.relu(self.bn1_a(self.conv1_a(x)))
        out_b = self.relu(self.bn1_b(self.conv1_b(x)))
        out_b, s2t_info = self.ssm(out_b)
        out_b = self.relu(out_b)
        out_a = self.tim(out_a)
        out_a = self.shift(out_a + self.ism(s2t_info))
        out_a = self.relu(self.k1(out_a))
        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TBRANCH(nn.Module):
    def __init__(self, num_segments, block, layers, groups=1, bottleneck_width=32,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd
        self.num_segments = num_segments

        super(TBRANCH, self).__init__()
        conv_layer = nn.Conv2d
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.num_segments, self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=1, is_first=is_first,
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.num_segments, self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=2, is_first=is_first,
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.num_segments, self.inplanes, planes,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def TB(num_segments, pretrained_path='scnet50_v1d-4109d1e1.pth', #path of scnet50_v1d-4109d1e1.pth
                **kwargs):
    model = TBRANCH(num_segments, TSBM, [3, 4, 6, 3],
                  deep_stem=True, stem_width=32, avg_down=True,
                  avd=True, **kwargs)
    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
    return model
