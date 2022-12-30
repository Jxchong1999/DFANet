import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


'''
Reference: 'Single-side domain generalization for face anti-spoofing' (CVPR'20)
- https://arxiv.org/abs/2004.14043
'''


def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)


class FWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = 128
        self.gamma1 = torch.nn.Parameter(torch.ones(1, self.c1, 1, 1) * 0.3)
        self.beta1 = torch.nn.Parameter(torch.ones(1, self.c1, 1, 1) * 0.5)

    def forward(self, x):
        if self.training:
            gamma = (1 + torch.randn(1, self.c1, 1, 1, device=self.gamma1.device) * softplus(self.gamma1)).expand_as(x)
            beta = (torch.randn(1, self.c1, 1, 1, device=self.beta1.device) * softplus(self.beta1)).expand_as(x)
            out = gamma * x + beta
            return out


class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        return out_normal


class LearnableAdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_y = nn.Parameter(torch.randn([128]), requires_grad=True)
        self.mu_y = nn.Parameter(torch.randn([128]), requires_grad=True)

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2,
                       (2, 3)) + 0.000000023) / (x.shape[2] * x.shape[3]))

    def forward(self, x):
        sigma_y = torch.randn(x.size()[:2]).cuda()
        mu_y = torch.randn(x.size()[:2]).cuda()
        # sigma_y==>[sigma_y,sigma_y,sigma_y] based on batch size
        for i in range(0, x.size(0)):
            sigma_y[i] = self.sigma_y
            mu_y[i] = self.mu_y
        return (sigma_y * ((x.permute([2, 3, 0, 1]) - self.mu(x)) /
                           self.sigma(x)) + mu_y).permute([2, 3, 0, 1])


class Ad_LDCNet(nn.Module):
    def __init__(self):
        super(Ad_LDCNet, self).__init__()
        self.Backbone = FE_Res18_learnable()
        self.LnessCsfier = Disentangled_Classifier(classes=2)
        self.DmainCsfier = Disentangled_Classifier(classes=3)
        self.FeDecoder = Feature_Decoder()
        self.LnAdaIN = LearnableAdaIN()
        self.FWT = FWT()  # Affine Feature Transform (AFT)

    def forward(self, input, update_step="Learn_Original"):
        if self.training:
            if update_step == "Learn_Original":
                # fixed adain
                self.Backbone.requires_grad = True
                self.LnessCsfier.requires_grad = True
                self.DmainCsfier.requires_grad = True
                self.FeDecoder.requires_grad = True
                self.LnAdaIN.requires_grad = False
                self.FWT.requires_grad = False
                # original
                catfeat = self.Backbone(input)
                # disentangled feature & prediction
                f_liveness, p_liveness = self.LnessCsfier(catfeat)
                f_liveness_norm = torch.norm(f_liveness, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_liveness = torch.div(f_liveness, f_liveness_norm)

                f_domain, p_domain = self.DmainCsfier(catfeat)
                f_domain_norm = torch.norm(f_domain, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain = torch.div(f_domain, f_domain_norm)

                # reconstruct original feature
                re_catfeat = self.FeDecoder(torch.cat([f_liveness, f_domain], 1))

                # unseen domain
                f_domain_hard = self.LnAdaIN(f_domain)
                f_domain_hard_norm = torch.norm(f_domain_hard, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain_hard = torch.div(f_domain_hard, f_domain_hard_norm)

                # reconstruct hard feature with unseen domain
                re_catfeat_hard = self.FeDecoder(torch.cat([f_liveness, f_domain_hard], 1))
                
                f_liveness_hard, p_liveness_hard = self.LnessCsfier(re_catfeat_hard)

                # diverse features
                f_liveness_fwt = self.FWT(f_liveness)
                f_liveness_fwt_norm = torch.norm(f_liveness_fwt, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (
                    2) ** 0.5
                f_liveness_fwt = torch.div(f_liveness_fwt, f_liveness_fwt_norm)

                # reconstruct diverse feature with seen domain
                re_catfeat_fwt = self.FeDecoder(torch.cat([f_liveness_fwt, f_domain], 1))
                
                f_liveness_fwt, p_liveness_fwt = self.LnessCsfier(re_catfeat_fwt)

                return catfeat, p_liveness, p_domain, re_catfeat, \
                       p_liveness_fwt, \
                       p_liveness_hard
            elif update_step == "Learn_FWT":
                # update Affine Feature Transform (AFT)
                self.Backbone.requires_grad = False
                self.LnessCsfier.requires_grad = False
                self.DmainCsfier.requires_grad = False
                self.FeDecoder.requires_grad = False
                self.LnAdaIN.requires_grad = False
                self.FWT.requires_grad = True

                catfeat = self.Backbone(input)
                # disentangled feature & prediction
                f_liveness, p_liveness = self.LnessCsfier(catfeat)
                f_liveness_norm = torch.norm(f_liveness, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_liveness = torch.div(f_liveness, f_liveness_norm)

                f_domain, p_domain = self.DmainCsfier(catfeat)
                f_domain_norm = torch.norm(f_domain, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain = torch.div(f_domain, f_domain_norm)

                # diverse features
                f_liveness_fwt = self.FWT(f_liveness)
                f_liveness_fwt_norm = torch.norm(f_liveness_fwt, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (
                    2) ** 0.5
                f_liveness_fwt = torch.div(f_liveness_fwt, f_liveness_fwt_norm)
                
                # reconstruct diverse feature with seen domain
                re_catfeat_fwt = self.FeDecoder(torch.cat([f_liveness_fwt, f_domain], 1))

                # diverse feature disentanglement
                f_liveness_fwt, p_liveness_fwt = self.LnessCsfier(re_catfeat_fwt)
                f_liveness_fwt_norm = torch.norm(f_liveness_fwt, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (
                    2) ** 0.5
                f_liveness_fwt = torch.div(f_liveness_fwt, f_liveness_fwt_norm)

                return p_liveness_fwt, f_liveness, f_liveness_fwt

            elif update_step == "Learn_Adain":
                # update Learnable adaIN
                self.Backbone.requires_grad = False
                self.LnessCsfier.requires_grad = False
                self.DmainCsfier.requires_grad = False
                self.FeDecoder.requires_grad = False
                self.LnAdaIN.requires_grad = True
                self.FWT.requires_grad = False

                catfeat = self.Backbone(input)
                # disentangled feature & prediction
                f_liveness, p_liveness = self.LnessCsfier(catfeat)
                f_domain, p_domain = self.DmainCsfier(catfeat)

                f_domain_norm = torch.norm(f_domain, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain = torch.div(f_domain, f_domain_norm)

                # unseen domain
                f_domain_hard = self.LnAdaIN(f_domain)
                f_domain_hard_norm = torch.norm(f_domain_hard, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (
                    2) ** 0.5
                f_domain_hard = torch.div(f_domain_hard, f_domain_hard_norm)

                re_catfeat_adain = self.FeDecoder(torch.cat([f_liveness, f_domain_hard], 1))
                
                f_liveness_hard, p_liveness_hard = self.LnessCsfier(re_catfeat_adain)
                return  p_liveness_hard


        else:
            catfeat = self.Backbone(input)
            f_liveness, p_liveness = self.LnessCsfier(catfeat)
            return p_liveness


class FE_Res18_learnable(nn.Module):
    def __init__(self):
        super(FE_Res18_learnable, self).__init__()

        model_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature1 = self.layer1(feature)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        re_feature1 = F.adaptive_avg_pool2d(feature1, 32)
        re_feature2 = F.adaptive_avg_pool2d(feature2, 32)
        re_feature3 = F.adaptive_avg_pool2d(feature3, 32)
        catfeat = torch.cat([re_feature1, re_feature2, re_feature3], 1)
        # L2 normalize
        feature_norm = torch.norm(catfeat, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        catfeat = torch.div(catfeat, feature_norm)

        return catfeat


class Feature_Decoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=448):
        super(Feature_Decoder, self).__init__()
        self.feature_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.feature_decoder(x)
        return x1


class Disentangled_Classifier(nn.Module):
    def __init__(self, in_channels=448, classes=2, conv3x3=conv3x3):
        super(Disentangled_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            conv3x3(128, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier_layer = nn.Linear(512, classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        classification = self.classifier_layer(x)

        return x1, classification




if __name__ == "__main__":
    net = Ad_LDCNet().cuda()
    net(torch.randn(2, 3, 256, 256).cuda())
