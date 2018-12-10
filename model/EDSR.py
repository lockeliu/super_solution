import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from model import common

class EDSR(nn.Module):
    def __init__(self, scale_list, model_path = 'weight/EDSR_weight.pt' ):
        super(EDSR, self).__init__()
        
        # args
        scale = scale_list[0]
        input_channel = 5 
        output_channel = 3
        num_block = 16 
        inp = 64
        rgb_range = 255 
        res_scale = 0.1
        act = nn.ReLU(True)
        #act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

        # head
        self.head = nn.Sequential( common.conv(3, inp, input_channel) )

        # body
        self.body = nn.Sequential( *[ common.ResBlock(inp, bias = True, act = act, res_scale = res_scale) for _ in range( num_block) ] )
        self.body.add_module( str(num_block), common.conv(inp, inp, 3) )
        
        # tail
        if scale > 1:
            self.tail = nn.Sequential( *[ common.Upsampler(scale, inp, act = False, choice = 0), 
                    common.conv(inp, 3, output_channel) ] )
        else:
            self.tail = nn.Sequential( *[ common.conv(inp, 3, output_channel) ] )

        self.sub_mean = common.MeanShift(rgb_range, sign = -1)
        self.add_mean = common.MeanShift(rgb_range, sign = 1)
        
        self.model_path = model_path 
        self.load()

    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def forward_pred( self, x, scale_id ):
        scale = torch.from_numpy( np.array( [scale_id] ) )
        return self.forward( x, scale )
    
    
    def _initialize_weights(self):
        for (name, m) in self.named_modules():
            if name.endswith('_mean'):
                print ('Do not initilize {}'.format(name) )
            elif isinstance(m, nn.Conv2d) and isinstance( m, nn.ReLU ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d) and isinstance( m, nn.LeakyReLU):
                nn.init.kaiming_normal_(m.weight, a = 0.05, mode = 'fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def savemodel(self):
        torch.save( self.state_dict(), self.model_path );

    def load(self):
        if os.path.exists(self.model_path):
            model = torch.load(self.model_path)
            self.load_state_dict(model)
        else:
            self._initialize_weights()
    
