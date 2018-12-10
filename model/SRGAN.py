import os
import sys
import torch
from model import common
import torch.nn as nn
import torch.nn.functional as F

class Generator( nn.Module ):
    def __init__(self, scale_list, model_path = 'weight/SRGAN_Generater_weight.pt'):
        super(Generator, self).__init__()
        
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
            elif isinstance(m, nn.Conv2d):
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

class Discriminator(nn.Module ):
    def __init__(self, model_path = 'weight/SRGAN_Discriminator_weight.pt' ):
        super(Discriminator, self).__init__()

        inp = 64

        self.body = nn.Sequential( 
                common.ConvBNSwish( 3, inp, 3, 1, bn = False),
                common.ConvBNSwish( inp, inp, 3, 2),
                common.ConvBNSwish( inp, inp * 2, 3, 1),
                common.ConvBNSwish( inp * 2, inp * 2, 3, 2),
                common.ConvBNSwish( inp * 2, inp * 4, 3, 1),
                common.ConvBNSwish( inp * 4, inp * 4, 3, 2),
                common.ConvBNSwish( inp * 4, inp * 8, 3, 1),
                common.ConvBNSwish( inp * 8, inp * 8, 3, 2),
                )

        self.fc = nn.Conv2d(inp * 8, 1, kernel_size=1, stride=1, padding=0)
        self.model_path = model_path
        self.load()

    def forward(self, x):
        x = self.body( x )

        x = self.fc(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


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
            elif isinstance(m, nn.Conv2d):
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

