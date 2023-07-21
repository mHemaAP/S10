import torch.nn as nn
import torchinfo
import torch.nn.functional as F


class convLayer(nn.Module):
    def __init__(self, l_input_c, 
                 l_output_c, bias=False, 
                 padding=1, stride=1, 
                 max_pooling=False, 
                 dropout=0):
        super (convLayer, self).__init__()


        self.convLayer = nn.Conv2d(in_channels=l_input_c, 
                          out_channels=l_output_c, 
                          kernel_size=(3, 3), 
                          stride=stride,
                          padding= padding,
                          padding_mode='replicate',
                          bias=bias)
        
        self.max_pooling = None
        if(max_pooling == True):
            self.max_pooling = nn.MaxPool2d(2, 2)

        self.normLayer = nn.BatchNorm2d(l_output_c)

        self.activationLayer = nn.ReLU()

        self.dropout = None
        if(dropout > 0):
            self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):

        x = self.convLayer(x)

        if (self.max_pooling is not None):
            x = self.max_pooling(x)        

        x = self.normLayer(x)
        x = self.activationLayer(x)
        
        if (self.dropout is not None):
            x = self.dropout(x)

        return x


class custBlock(nn.Module):
    def __init__(self, l_input_c, 
                 l_output_c, bias=False, 
                 padding=1, stride=1, 
                 max_pooling=True, 
                 dropout=0, residual_links=2):
        super (custBlock, self).__init__()


        self.conv_pool_block = convLayer(l_input_c=l_input_c,
                                l_output_c=l_output_c, 
                                bias=bias, padding=padding,
                                stride=stride, max_pooling=max_pooling, 
                                dropout=dropout)
        
        self.residual_block = None
        if(residual_links > 0):
            res_layer_seq = []
            for link in range(0, residual_links):
                res_layer_seq.append(
                            convLayer(l_input_c=l_output_c,
                                l_output_c=l_output_c, 
                                bias=bias, padding=padding,
                                stride=stride, max_pooling=False, 
                                dropout=dropout)                    
                )

            self.residual_block = nn.Sequential(*res_layer_seq)                       

    
    def forward(self, x):

        x = self.conv_pool_block(x)

        if (self.residual_block is not None):
            tmp_x = x
            x = self.residual_block(x)
            x = x +  tmp_x   

        return x


class custResNet(nn.Module):
    def __init__(self, dropout=0):
        super(custResNet, self).__init__()

        ##### Prep Block #####
        # This block has a 3x3 convolution layer with stride=1, 
        # padding=1 followed by batch normalization and RELU
        # 64 kernels are used in this block
        # By default, dropout is set to 0
        self.prep_block = custBlock(l_input_c=3, l_output_c=64, 
                                    max_pooling=False, dropout= dropout,
                                    residual_links=0
                                    ) # output_size = 32, rf_out = 3 
        
        ##### Convolution Block - 1 #####
        # This block in the first step has a 3x3 convolution layer with 
        # stride=1, padding=1 followed by Max pooling, batch normalization
        # and ReLU. In the second step a network with residual links, with 
        # each link having 3x3 convolution with stride=1, padding=1 
        # followed by batch normalization and ReLU. And in the third step,
        # the result from the first step and the result of the residual network  
        # from the second step are added to make a skip connection.
        # 128 kernels are used in this block
        # By default, dropout is set to 0
        self.block1 = custBlock(l_input_c=64, l_output_c=128, 
                                max_pooling=True, dropout= dropout,
                                residual_links=2
                                ) # output_size = 16, rf_out = 13 

        ##### Convolution Block - 2 #####
        # This block in the first step has a 3x3 convolution layer with 
        # stride=1, padding=1 followed by Max pooling, batch normalization
        # and ReLU. 
        # 256 kernels are used in this block
        # By default, dropout is set to 0        
        self.block2 = custBlock(l_input_c=128, l_output_c=256, 
                                max_pooling=True, dropout= dropout,
                                residual_links=0
                                ) # output_size = 8, rf_out = 17

        ##### Convolution Block - 3 #####
        # This block in the first step has a 3x3 convolution layer with 
        # stride=1, padding=1 followed by Max pooling, batch normalization
        # and ReLU. In the second step a network with residual links, with 
        # each link having 3x3 convolution with stride=1, padding=1 
        # followed by batch normalization and ReLU. And in the third step,
        # the result from the first step and the result of the residual network  
        # from the second step are added to make a skip connection.
        # 512 kernels are used in this block
        # By default, dropout is set to 0        
        self.block3 = custBlock(l_input_c=256, l_output_c=512, 
                                max_pooling=True, dropout= dropout,
                                residual_links=2
                                ) # output_size = 4, rf_out = 57 

        self.max_pool_layer = nn.MaxPool2d(4, 4)
        # output_size = 1, rf_out = 81
        self.flatten_layer = nn.Flatten()
        self.fc = nn.Linear(512, 10)
        #self.softmax = nn.Softmax()       


    def forward(self, x):

        x = self.prep_block(x)
        x = self.block1(x)        
        x = self.block2(x)
        x = self.block3(x)
        x = self.max_pool_layer(x)
        x = self.flatten_layer(x)        
        x = self.fc(x)

        return x 

    
    # Network Summary
    def summary(self, input_size=None, depth=10):
        return torchinfo.summary(self, input_size=input_size,
                                 depth=depth,
                                 col_names=["input_size", 
                                            "output_size", 
                                            "num_params",
                                            "kernel_size", 
                                            "params_percent"])     
        
