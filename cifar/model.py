import torch.nn as nn
import torchinfo
import torch.nn.functional as F
#from torchsummary import summary


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

# class convLayer(nn.Module):
#     def __init__(self, l_input_c, 
#                  l_output_c, bias=False, 
#                  padding=1, stride=1, 
#                  max_pooling=False, 
#                  dropout=0):
#         super (convLayer, self).__init__()

#         sub_layers = []

#         sub_layers.append(nn.Conv2d(in_channels=l_input_c, 
#                           out_channels=l_output_c, 
#                           kernel_size=(3, 3), 
#                           stride=stride,
#                           padding= padding,
#                           padding_mode='replicate',
#                           bias=bias))
        
        
#         if(max_pooling == True):
#             sub_layers.append(nn.MaxPool2d(2, 2))

#         sub_layers.append(nn.BatchNorm2d(l_output_c))

#         sub_layers.append(nn.ReLU())

#         if(dropout > 0):
#             sub_layers.append(nn.Dropout(dropout))

#         self.layer = nn.Sequential(*sub_layers)
    
#     def forward(self, x):

#         return self.layer(x)

# class custBlock(nn.Module):
#     def __init__(self, input_c, output_c, pool=True, residue=2, dropout=0):
#         super(custBlock, self).__init__()

#         self.pool_block = convLayer(input_c, output_c, max_pooling=pool, dropout=dropout)
#         self.res_block = None
#         if residue > 0:
#             layers = list()
#             for i in range(0, residue):
#                 layers.append(
#                     convLayer(output_c, output_c, max_pooling=False, dropout=dropout)
#                 )
#             self.res_block = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.pool_block(x)
#         if self.res_block is not None:
#             x_ = x
#             x = self.res_block(x)
#             # += operator causes inplace errors in pytorch if done right after relu.
#             x = x + x_
#         return x



class custResNet(nn.Module):
    def __init__(self, dropout=0):
        super(custResNet, self).__init__()


        self.prep_block = custBlock(l_input_c=3, l_output_c=64, 
                                    max_pooling=False, dropout= dropout,
                                    residual_links=0
                                    ) # output_size = , rf_out = 
        

        self.block1 = custBlock(l_input_c=64, l_output_c=128, 
                                max_pooling=True, dropout= dropout,
                                residual_links=2
                                ) # output_size = , rf_out = 
        
        self.block2 = custBlock(l_input_c=128, l_output_c=256, 
                                max_pooling=True, dropout= dropout,
                                residual_links=0
                                ) # output_size = , rf_out =
        
        self.block3 = custBlock(l_input_c=256, l_output_c=512, 
                                max_pooling=True, dropout= dropout,
                                residual_links=2
                                ) # output_size = , rf_out = 

        self.max_pool_layer = nn.MaxPool2d(4, 4)
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

        return F.log_softmax(x, dim=1)

# class custResNet(nn.Module):
#     def __init__(self, dropout=0):
#         super(custResNet, self).__init__()


#         self.prep_block = custBlock(input_c=3, output_c=64, 
#                                          pool=False, dropout= dropout,
#                                          residue=0
#                                          ) # output_size = , rf_out = 
        

#         self.block1 = custBlock(input_c=64, output_c=128, 
#                                          pool=True, dropout= dropout,
#                                          residue=2
#                                          ) # output_size = , rf_out = 
        
#         self.block2 = custBlock(input_c=128, output_c=256, 
#                                          pool=True, dropout= dropout,
#                                          residue=0
#                                          ) # output_size = , rf_out =
        
#         self.block3 = custBlock(input_c=256, output_c=512, 
#                                          pool=True, dropout= dropout,
#                                          residue=2
#                                          ) # output_size = , rf_out = 

#         self.max_pool_layer = nn.MaxPool2d(4, 4)
#         self.flatten_layer = nn.Flatten()
#         self.fc = nn.Linear(512, 10)
#         #self.softmax = nn.Softmax()       


#     def forward(self, x):

#         x = self.prep_block(x)
#         x = self.block1(x)        
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.max_pool_layer(x)
#         x = self.flatten_layer(x)        
#         x = self.fc(x)

#         return F.log_softmax(x, dim=1)
    
    # Network Summary
    def summary(self, input_size=None, depth=10):
        return torchinfo.summary(self, input_size=input_size,
                                 depth=depth,
                                 col_names=["input_size", 
                                            "output_size", 
                                            "num_params",
                                            "kernel_size", 
                                            "params_percent"])     
        
