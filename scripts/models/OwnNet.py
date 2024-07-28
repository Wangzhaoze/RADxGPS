# All rights reserved. 

# Copyright (c) 2020

# Source and binary forms are subject non-exclusive, revocable, non-transferable, and limited right to use the code for the exclusive purpose of undertaking academic or not-for-profit research.

# Redistributions must retain the above copyright notice, this license and the following disclaimer.

# Use of the code or any part thereof for commercial purposes is strictly prohibited.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#from imports import *
from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(),min=1e-4,max=1-1e-4)
    return y

class spatial_attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(F_g,F_int,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int))
        self.w_x = nn.Sequential(nn.Conv2d(F_l,F_int,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int,1,kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(1),
        nn.Sigmoid())     
        self.relu = nn.ReLU(inplace=True)
    def forward(self,g,x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(x1+g1)
        psi = self.psi(psi)

        return x*psi  

class add_coord(nn.Module):
    
    def __init__(self,image_shape,gpus_list,batch_size):
        super().__init__()
        image_height = image_shape[0]
        image_width  = image_shape[1]
               
        y_coords = (2.0 * torch.arange(image_height).unsqueeze(1).expand(image_height, image_width) / (image_height - 1.0) - 1.0)
        x_coords = torch.arange(image_width).unsqueeze(0).expand(image_height, image_width).float() / image_width
        self.coords = torch.unsqueeze(torch.stack((y_coords, x_coords), dim=0), dim=0).repeat(batch_size, 1, 1, 1)
        self.coords = self.coords.cuda(gpus_list[0])
        
    def forward(self,x):
        return torch.cat((x,self.coords), dim=1)

class CAM_Module(nn.Module):
    def __init__(self, in_dim,batch_size):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma     = nn.Parameter(torch.ones(batch_size,in_dim,1,1))
        self.softmax   = nn.Softmax(dim=-1)
        
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key   = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy     = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention  = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        return self.gamma*out + x

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch,batch_size,kernel_size=3,channel_attention=1):
        super().__init__()
        self.channel_attention = channel_attention
        self.conv1  = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.norm1  = nn.InstanceNorm2d(out_ch,affine=True)
        #self.norm1  = nn.BatchNorm2d(out_ch,affine=True)
        self.atten1 = CAM_Module(in_ch,batch_size)
        self.act1   = nn.LeakyReLU()
        
        self.conv2  = nn.Conv2d(in_ch+out_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.norm2  = nn.InstanceNorm2d(out_ch,affine=True)
        #self.norm2  = nn.BatchNorm2d(out_ch,affine=True)
        self.atten2 = CAM_Module(in_ch+out_ch,batch_size)
        self.act2   = nn.LeakyReLU()

    def forward(self, x):
        res = x
        if self.channel_attention:
            x = self.atten1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        if self.channel_attention==2:        
            x = self.atten2(torch.cat([res,x],dim=1))
        else:    
            x = torch.cat([res,x],dim=1)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,batch_size,channel_att=1,kernel_size=3):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch,batch_size,kernel_size,channel_att)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch,batch_size,max_pooling=False,channel_att=1,kernel_size=3):
        super().__init__()
        if max_pooling:
            self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch,batch_size,kernel_size,channel_att))
        else:    
            self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            double_conv(in_ch, out_ch,batch_size,kernel_size,channel_att))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch,batch_size, bilinear=False,spatial_attention=False,channel_att=1,swap =False,conv_swap_in=0,kernel_size=3):
        super().__init__()
        self.swap = swap
        if swap == True:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=(1,2), stride=(1,2))

            self.conv = double_conv(in_ch+conv_swap_in, out_ch,batch_size,kernel_size,channel_att)
            self.spatial_attention = spatial_attention
            if spatial_attention:
                self.s_att = spatial_attention_block(F_g=in_ch,F_l=conv_swap_in,F_int=int(conv_swap_in/2))
        else:    
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

            self.conv = double_conv(in_ch, out_ch,batch_size,kernel_size,channel_att)
            self.spatial_attention = spatial_attention
            if spatial_attention:
                self.s_att = spatial_attention_block(F_g=int(in_ch/2),F_l=int(in_ch/2),F_int=int(in_ch/4))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # if diffX<0:
        #     diffX = abs(diffX)
        #     if diffY<0:
        #         diffY = abs(diffY)
        #     x1 = F.pad(x1, (diffY // 2, diffY - diffY//2, diffX // 2, diffX - diffX//2))
        # elif diffX>0:
        #     x2 = F.pad(x2, (diffY // 2, diffY - diffY//2, diffX // 2, diffX - diffX//2))
        # elif diffX==0:
        #     x2 = F.pad(x2, (diffY // 2, diffY - diffY//2, diffX // 2, diffX - diffX//2))
        if self.spatial_attention:
            x2 = self.s_att(x1,x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

def _get_deconv_cfg(deconv_kernel):
    if deconv_kernel ==4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0

    return deconv_kernel, padding, output_padding    


class header(nn.Module):
    def __init__(self,in_ch,middle_ch,out_ch,norm=False):
        super().__init__() 
        self.norm = norm
        self.conv1 = nn.Conv2d(in_ch,middle_ch,kernel_size=3,padding=1,bias=True)      
        if (self.norm and self.norm=='batchnorm'):
            
            self.norm1  = nn.BatchNorm2d(middle_ch,affine=True)
        elif (self.norm and self.norm =='instancenorm') :
            self.norm1  = nn.InstanceNorm2d(middle_ch,affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=True)
    def forward(self,x):
        x = self.conv1(x)
        if self.norm :
            x=self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        return x   

class mimo_encoder(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=(12,1),dilation=(16,1),use_bn = False):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size,stride=(1,1),padding=0,dilation=dilation,bias=(not use_bn))
        self.bn1 = nn.BatchNorm2d(out_ch)
        #self.bn1 = nn.InstanceNorm2d(out_ch)
        self.padding = int(NbVirtualAntenna/2)

    def forward(self,x):
        width = x.shape[-1-1] 
        x = torch.concat([x[:,:,-self.padding:,:] ,x,x[:,:,:self.padding,:]],dim=2  )
        x = self.conv1(x)
        x = x[:,:,int(x.shape[-1-1]/2-width/2):int(x.shape[-1-1]/2+width/2),: ] 
        if self.use_bn==True:
            x = self.bn1(x)

        return x    

class ISM_Net(nn.Module):
    def __init__(self,gpus_list,batch_size,output_shape,num_classes,mimo_encoder_bn=False,seg_branch_spatial_att=True,max_pooling_down=False,encoder_channel_att=1,decoder_channel_att=1,scaleFactor=2,swap=False,encoder_kernelsize=7,decoder_kernelsize=7):
        super().__init__()
        self.swap = swap#swap azi and doppler
        #self.simp_det_header = simp_det_header
        self.mimo_layer = mimo_encoder(16*2,NbVirtualAntenna,use_bn=mimo_encoder_bn)
        #self.inc    = inconv(48*2+2,               int(64/scaleFactor),batch_size)
        self.inc    = inconv(NbVirtualAntenna+2,   int(64/scaleFactor),batch_size,channel_att=encoder_channel_att,kernel_size=encoder_kernelsize)
        #self.inc    = inconv(NbVirtualAntenna,   int(64/scaleFactor),batch_size)
        self.down1  = down(int(64/scaleFactor),    int(128/scaleFactor),batch_size,max_pooling=max_pooling_down,channel_att=encoder_channel_att,kernel_size=encoder_kernelsize)
        self.down2  = down(int(128/scaleFactor),   int(256/scaleFactor),batch_size,max_pooling=max_pooling_down,channel_att=encoder_channel_att,kernel_size=encoder_kernelsize)
        self.down3  = down(int(256/scaleFactor),   int(512/scaleFactor),batch_size,max_pooling=max_pooling_down,channel_att=encoder_channel_att,kernel_size=encoder_kernelsize)
        self.down4  = down(int(512/scaleFactor),   int(512/scaleFactor),batch_size,max_pooling=max_pooling_down,channel_att=encoder_channel_att,kernel_size=encoder_kernelsize)

        if self.swap == False:
            self.up1    = up(int(1024/scaleFactor),    int(256/scaleFactor),batch_size,spatial_attention=seg_branch_spatial_att,channel_att=decoder_channel_att,kernel_size=decoder_kernelsize)
            self.up2    = up(int(512/scaleFactor),     int(128/scaleFactor),batch_size,spatial_attention=seg_branch_spatial_att,channel_att=decoder_channel_att,kernel_size=decoder_kernelsize)
            self.up3    = up(int(256/scaleFactor),     int(64/scaleFactor),batch_size,spatial_attention=seg_branch_spatial_att,channel_att=decoder_channel_att,kernel_size=decoder_kernelsize)
            self.up4    = up(int(128/scaleFactor),     int(64/scaleFactor),batch_size,spatial_attention=seg_branch_spatial_att,channel_att=decoder_channel_att,kernel_size=decoder_kernelsize)
       
            self.outc   = outconv(int(64/scaleFactor), 1) #num_classes 2: occupancy vs free
        else:
            self.up1 = up(16, 48,batch_size,spatial_attention=seg_branch_spatial_att,channel_att=decoder_channel_att,swap =self.swap ,conv_swap_in=32,kernel_size=decoder_kernelsize)
            self.up2 = up(48,128,batch_size,spatial_attention=seg_branch_spatial_att,channel_att=decoder_channel_att,swap =self.swap ,conv_swap_in=64,kernel_size=decoder_kernelsize)
            self.up3 = up(128, 256,batch_size,spatial_attention=seg_branch_spatial_att,channel_att=decoder_channel_att,swap =self.swap ,conv_swap_in=128,kernel_size=decoder_kernelsize)
            self.up4 = up(256,256,batch_size,spatial_attention=seg_branch_spatial_att,channel_att=decoder_channel_att,swap =self.swap ,conv_swap_in=256,kernel_size=decoder_kernelsize)
            self.conv1x1_x3 = nn.Conv2d(int(256/scaleFactor),256,kernel_size=1)
            self.conv1x1_x2 = nn.Conv2d(int(128/scaleFactor),256,kernel_size=1)
            self.conv1x1_x1 = nn.Conv2d(int(64/scaleFactor),256,kernel_size=1)

            self.outc1 = double_conv(256, 128,batch_size,kernel_size=decoder_kernelsize,channel_attention=decoder_channel_att)
            self.outc2 = double_conv(128,64,batch_size,kernel_size=decoder_kernelsize,channel_attention=decoder_channel_att)
            self.outc3 = outconv(64,1)

        # Position encoding
        self.coord = add_coord(output_shape,gpus_list,batch_size)


    def forward(self, x):
        x0 = self.mimo_layer(x)        
        x1 = self.inc(self.coord(x0))
        #x1 = self.inc(x0)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        out ={}
        if self.swap==False:
            x  = self.up1(x5, x4)
            x  = self.up2(x, x3)
            x  = self.up3(x, x2)
            x  = self.up4(x, x1)
            x  = self.outc(x)
        else:    
            x = self.up1(x5.transpose(1,2),x4.transpose(1,2))
            x = self.up2(x,self.conv1x1_x3(x3).transpose(1,2))
            x = self.up3(x,self.conv1x1_x2(x2).transpose(1,2))
            x = self.up4(x,self.conv1x1_x1(x1).transpose(1,2))
            x = self.outc1(x)
            x = self.outc2(x)
            x = self.outc3(x)

        out['seg_header'] = x
                 
        return out
