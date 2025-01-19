import torch
import torch.nn as nn
import torch.nn.functional as F  
# import torchvision.models as models
# from torchvision.models import resnet50, ResNet50_Weights
import timm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MainBranch(nn.Module): # Main branch用SwinT，輸入是原始圖像
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True): 
        super(MainBranch, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = Identity() # 去掉最後一層全連接層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 添加GAP

    def forward(self, x):
        x = self.model(x)
    
        if x.shape[1] != 3: # 如果模型輸出順序是NHWC，轉為NCHW
            x = x.permute(0, 3, 1, 2)
            
        x = self.avgpool(x)
        return x


class AuxBranch(nn.Module): # 輸入是原始圖像的DC map和Saturation map
    def __init__(self, in_channel=2):
        super(AuxBranch, self).__init__()
        condition_conv1 = nn.Conv2d(in_channel, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        condition_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        condition_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)

        conditon_conv = [
            condition_conv1, nn.LeakyReLU(0.2, True),
            condition_conv2, nn.LeakyReLU(0.2, True),
            condition_conv3, nn.LeakyReLU(0.2, True)
        ]
        self.condition_conv = nn.Sequential(*conditon_conv)

        sift_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        sift_conv2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        sift_conv = [
            sift_conv1, nn.LeakyReLU(0.2, True),
            sift_conv2, nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1))
        ]
        self.sift_conv = nn.Sequential(*sift_conv)

    def forward(self, x):
        x = self.condition_conv(x)
        x = self.sift_conv(x)
        return x

class GatingWithRH(nn.Module):
    def __init__(self, feature_dim=512, rh_dim=1):
        super(GatingWithRH, self).__init__()
        self.gate = nn.Linear(rh_dim, feature_dim)

    def forward(self, aux_output, rh):
        gate_values = torch.sigmoid(self.gate(rh))

        gate_values = gate_values.unsqueeze(-1).unsqueeze(-1)

        gate_values = gate_values.expand_as(aux_output)
        gated_features = aux_output * gate_values  # 逐元素相乘
        return gated_features

        

class SE_block1D(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SE_block1D, self).__init__()
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.ca(x)
        return x * y


class Fusion_Block(nn.Module):
    def __init__(self, in_channel):
        super(Fusion_Block, self).__init__()
        self.fc1 = nn.Conv2d(in_channel, 512, 1, padding=0, bias=True)
        self.se = SE_block1D(512, reduction=8)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 1),
        )

    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=1)
        y = self.fc1(cat)
        y = self.se(y)
        y = torch.flatten(y, 1)
        y = self.out(y)
        return y

class CompleteModel3(nn.Module):
    def __init__(self):
        super(CompleteModel3, self).__init__()
        self.main_branch = MainBranch(model_name='swin_tiny_patch4_window7_224', pretrained=True)
        self.aux_branch = AuxBranch(in_channel=2)      
        self.gate_branch = GatingWithRH(feature_dim=512, rh_dim=1)     
        self.fusion_block = Fusion_Block(in_channel=768 + 512)  # Main branch + Attention Branch

    def forward(self, main_input, aux_input, RH):
        main_out = self.main_branch(main_input)
        aux_out = self.aux_branch(aux_input)
        gate_out = self.gate_branch(aux_out, RH)

        # print("Shape of main_out:", main_out.shape)
        # print("Shape of gate_out:", gate_out.shape)
        
        output = self.fusion_block(main_out, gate_out)
        return output

    
if __name__ == "__main__":
    model = CompleteModel3()
    
    # 模擬
    main_input = torch.randn(1, 3, 224, 224)
    aux_input = torch.randn(1, 2, 224, 224)
    RH = torch.tensor([[0.85]])

    outputs = model(main_input, aux_input, RH)
    print(outputs)
