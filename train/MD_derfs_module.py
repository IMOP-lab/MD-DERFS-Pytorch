import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init,BCEWithLogitsLoss



def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc
def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]
def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


##SDE层定义
class MFU_module(nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super(MFU_module, self).__init__()
        self.inter_channels = in_channels // 8

        self.att1 = DANetHead(self.inter_channels,self.inter_channels)
        self.att2 = DANetHead(self.inter_channels,self.inter_channels)
        self.att3 = DANetHead(self.inter_channels,self.inter_channels)
        self.att4 = DANetHead(self.inter_channels,self.inter_channels)
        self.att5 = DANetHead(self.inter_channels,self.inter_channels)
        self.att6 = DANetHead(self.inter_channels,self.inter_channels)
        # self.att7 = DANetHead(self.inter_channels,self.inter_channels)
        # self.att8 = DANetHead(self.inter_channels,self.inter_channels)


        self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, out_channels, 1))
        #self.encoder_block = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 32, 1))
        
        if num_class<32:
            self.reencoder = nn.Sequential(
                        nn.Conv2d(num_class, num_class*8, 1),
                        nn.ReLU(True),
                        nn.Conv2d(num_class*8, in_channels, 1))
        else:
            self.reencoder = nn.Sequential(
                        nn.Conv2d(num_class, in_channels, 1),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels, in_channels, 1))
            
        self.iaff = iAFF(channels=64)

    def forward(self, x, d_prior):


        enc_feat = self.reencoder(d_prior)

        feat1 = self.att1(x[:,:self.inter_channels], enc_feat[:,0:self.inter_channels])
        feat2 = self.att2(x[:,self.inter_channels:2*self.inter_channels],enc_feat[:,self.inter_channels:2*self.inter_channels])
        feat3 = self.att3(x[:,2*self.inter_channels:3*self.inter_channels],enc_feat[:,2*self.inter_channels:3*self.inter_channels])
        feat4 = self.att4(x[:,3*self.inter_channels:4*self.inter_channels],enc_feat[:,3*self.inter_channels:4*self.inter_channels])
        feat5 = self.att5(x[:,4*self.inter_channels:5*self.inter_channels],enc_feat[:,4*self.inter_channels:5*self.inter_channels])
        feat6 = self.att6(x[:,5*self.inter_channels:6*self.inter_channels],enc_feat[:,5*self.inter_channels:6*self.inter_channels])


        # 6DA
        feat_aff1 = self.iaff(feat1, feat2)
        feat_aff2 = self.iaff(feat3, feat4)
        feat_aff3 = self.iaff(feat5, feat6)
        feat = torch.cat([feat_aff1,feat_aff2,feat_aff3,feat_aff1,feat_aff2,feat_aff3,feat1,feat6],dim=1)

        sasc_output = self.final_conv(feat)
        sasc_output = sasc_output+x

        return sasc_output
##DCNetHead定义
class DANetHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        # inter_channels = in_channels // 8
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = ACmix(inter_channels, inter_channels)
        self.sc = ACmix(inter_channels, inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))

    def forward(self, x, enc_feat):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)
        
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        feat_sum = feat_sum*F.sigmoid(enc_feat)

        sasc_output = self.conv8(feat_sum)


        return sasc_output
    
class ChannelAdjustment(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAdjustment, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(args.device)


    def forward(self, x):
        # 使用卷积层将通道数从 in_channels 变换为 out_channels
        return self.conv(x)


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class MLFABlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d,scale=2,relu=True,last=False):
        super(MLFABlock, self).__init__()
       

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
       
        self.scale=scale
        self.last=last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        
        if self.last==False:
            x = self.conv_3x3(x)
        if self.scale>1:
            x=F.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=True)
        x = x.to('cuda')
        #print(x.dtype)
        x=self.conv_1x1(x)   #x: torch.Size([1, 64, 512, 512])
        # print("x:",x.shape)
        return x
    

class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3*self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride)

        self.reset_parameters()
    
    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i//self.kernel_conv, i%self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h//self.stride, w//self.stride

        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, h, w)
        v_att = v.view(b*self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # 1, head_dim, k_att^2, h_out, w_out
        
        att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1) # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h*w), k.view(b, self.head, self.head_dim, h*w), v.view(b, self.head, self.head_dim, h*w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        
        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv
    


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo








class AADBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_in,
                 out_channels,
                 scale_aware_proj=False):
        super(AADBlock, self).__init__()
        self.scale_aware_proj = scale_aware_proj


        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

        self.content_encoders=nn.Sequential(
                nn.Conv2d(channel_in, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        
        self.feature_reencoders=nn.Sequential(
                nn.Conv2d(channel_in, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        

        self.normalizer = nn.Sigmoid()

        # Spatial Pyramid Pooling (SPP)
        self.spp = nn.ModuleList([
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=9, stride=1, padding=4),
            nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        ])
        self.spp_merge = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, scene_feature, features):
        # Content Features
        content_feats = self.content_encoders(features)

        # Scene Features
        scene_feat = self.scene_encoder(scene_feature)

        # Apply Spatial Pyramid Pooling (SPP) to the scene feature
        spp_feats = [scene_feat]
        for pool in self.spp:
            spp_feats.append(pool(scene_feat))
        spp_feats = torch.cat(spp_feats, dim=1)
        scene_feat = self.spp_merge(spp_feats)

        # Calculating Relations
        relations = self.normalizer((scene_feat * content_feats).sum(dim=1, keepdim=True))

        # Feature Re-encoding
        p_feats = self.feature_reencoders(features)

        # Refined Features
        refined_feats = relations * p_feats

        return refined_feats