import torch.nn as nn
from .cnn_pred import Cnn_predNet, Cnn_featspr
from .gate_inference import Inference_gate
from .cnn_backbone.getnet import get_backnet

class FDGRNet(nn.Module):
    def __init__(self, backbone_name, output_shape, num_class, pretrained=True):
        super(FDGRNet, self).__init__()
        channel_settings,  self.backbone = get_backnet(backbone_name, pretrained=pretrained)
        self.cnn_prednet = Cnn_predNet(channel_settings)
        self.cnn_fs = Cnn_featspr(channel_settings, output_shape, num_class)
        self.gate_inference = Inference_gate(num_class)

    def forward(self, x):
        back_out = self.backbone(x)
        fea_l = self.cnn_prednet(back_out)
        cnn_feas, cnn_outs = self.cnn_fs(x, fea_l)
        gcn_outs = self.gate_inference(cnn_feas)
        return cnn_outs, gcn_outs
