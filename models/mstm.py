import torch.nn as nn

try:
    from models.tb import *
    from models.vit_torch import *
except:
    from tb import *
    from vit_torch import *


class MSTM(nn.Module):
    def __init__(self,
                 num_class=2,
                 num_segment=8,
                 add_softmax=True,
                 ):
        super().__init__()

        self.num_class = num_class
        self.num_segment = num_segment
        self.add_softmax = add_softmax
        self.build_model1()
        self.build_model2()
        self.weight_layer = nn.Linear(4, 2)

    def build_model1(self):
        self.base_model1 = TB(self.num_segment)
        fc_feature_dim = self.base_model1.fc.in_features
        self.base_model1.fc = nn.Linear(fc_feature_dim, self.num_class)
        if self.add_softmax:
            self.softmax_layer = nn.Softmax(dim=1)

    def build_model2(self):
        # self.base_model2 = vit_base_patch16_224_in21k(2,True)
        # self.base_model2 = vit_base_patch32_224_in21k(2, True)
        self.base_model2 = vit_large_patch16_224_in21k(2, True)
        # self.base_model2 = vit_large_patch32_224_in21k(2, True)

    def forward(self, x):
        img_channel = 3
        x_t, x_p = torch.split(x, [x.size(1) - 3, 3], dim=1)
        out = self.base_model1(x_t.reshape((-1, img_channel) + x.size()[2:]))
        out = out.view(-1, self.num_segment, self.num_class)
        out1 = out.mean(1, keepdim=False)

        out2 = self.base_model2(
            x_p.view((-1, img_channel) + x_p.size()[2:]).permute(0, 2, 3, 1)
        )

        if self.add_softmax:
            out1 = self.softmax_layer(out1)
            out2 = self.softmax_layer(out2)
        outlast = out1 * 0.7 + out2 * 0.3
        return outlast

    def set_segment(self, num_segment):
        self.num_segment = num_segment
