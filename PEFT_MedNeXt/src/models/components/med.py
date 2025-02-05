import torch

#Model
class MedNext(torch.nn.Module):
    def __init__(self,in_channels,out_channels,exp_r):
        super().__init__()
        self.exp_r=exp_r
        self.conv1=torch.nn.Conv3d(in_channels,in_channels,7,1,(3,3,3),groups=in_channels)
        #using groups=in_channels to decrease the number of the trainable parameter, making model light weight
        self.act=torch.nn.GELU()
        self.conv2=torch.nn.Conv3d(in_channels,exp_r*in_channels,1,1,0)
        self.norm=torch.nn.GroupNorm(num_groups=in_channels,num_channels=exp_r*in_channels)
        self.conv3=torch.nn.Conv3d(exp_r*in_channels,in_channels,1,1,0)
    
    def forward(self,x):
        x1=self.conv1(x)
        x1=self.act(x1)
        x1=self.conv2(x1)
        x1=self.norm(x1)
        x1=self.act(x1)
        x1=self.conv3(x1)
        
        return x1

class MedDownSample(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,exp_r):
        super().__init__()     
        self.exp_r=exp_r
        
        self.conv1=torch.nn.Conv3d(in_channels,in_channels,7,2,(3,3,3),groups=in_channels)
        
        #using groups=in_channels to decrease the number of the trainable parameter, making model light weight
        self.act=torch.nn.GELU()
        self.conv2=torch.nn.Conv3d(in_channels,(exp_r*in_channels),1,1,0)
        self.norm=torch.nn.GroupNorm(num_groups=in_channels,num_channels=exp_r*in_channels)
        
        self.conv3=torch.nn.Conv3d(exp_r*in_channels,(2*in_channels),1,1,0)
        
        
        #MedNext DownSample
        self.do_res=True
        
        if self.do_res:
            self.res_conv1=torch.nn.Conv3d(in_channels,(2*in_channels),1,2)
        
    def forward(self,x):
        x1=self.conv1(x)
        x1=self.act(x1)
        x1=self.conv2(x1)
        x1=self.norm(x1)
        x1=self.act(x1)
        x1=self.conv3(x1)

        res=self.res_conv1(x)
        if self.do_res:
            res=self.res_conv1(x)
            x1=x1+res
        return x1

class MedUpSample(torch.nn.Module):
    def __init__(self,in_channels,out_channels,exp_r):
        super().__init__()     
        self.conv1=torch.nn.ConvTranspose3d(in_channels,in_channels,7,2,(3,3,3),groups=in_channels)
        #using groups=in_channels to decrease the number of the trainable parameter, making model light weight
        self.act=torch.nn.GELU()
        self.conv2=torch.nn.Conv3d(in_channels,exp_r*in_channels,1,1,0)
        self.norm=torch.nn.GroupNorm(num_groups=in_channels,num_channels=exp_r*in_channels)
        self.conv3=torch.nn.Conv3d(exp_r*in_channels,int(in_channels/2),1,1,0)
        #MedNext DownSample
        self.do_res=True

        if self.do_res:
            self.res_conv1=torch.nn.ConvTranspose3d(in_channels,int(in_channels/2),1,2)
        
    def forward(self,x):
        x1=self.conv1(x)
        x1=self.act(x1)
        x1=self.conv2(x1)
        x1=self.norm(x1)
        x1=self.act(x1)
        x1=self.conv3(x1)
        x1=torch.nn.functional.pad(x1,(1,0,1,0,1,0))

        if self.do_res:
            res=self.res_conv1(x)
            res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
            x1=x1+res
        return x1


class Med_Final2(torch.nn.Module):
    def __init__(self,in_channels,out_channels,exp_rate=[2,3,4,4,4,4,4,3,2],num_blocks=[2,2,2,2,2,2,2,2,2]):
        super().__init__()
        
        self.exp_rate=exp_rate
        self.num_blocks=num_blocks
        
        self.semm=torch.nn.Conv3d(in_channels,in_channels,1)
        
        self.med_1=torch.nn.Sequential(*[
            MedNext(in_channels,in_channels,exp_rate[0])
            for i in range(num_blocks[0])
        ])
        
        self.encoder_down1= MedDownSample(in_channels,out_channels,exp_rate[1])
        
        self.med_2=torch.nn.Sequential(*[
            MedNext(2*in_channels,in_channels,exp_rate[1])
            for i in range(self.num_blocks[1])
        ])
        
        
        self.encoder_down2=MedDownSample(2*in_channels,out_channels,exp_rate[2])
        
        self.med_3=torch.nn.Sequential(*[
            MedNext(4*in_channels,out_channels,exp_rate[2])
            for i in range(self.num_blocks[2])
        ])
        
        self.encoder_down3=MedDownSample(4*in_channels,out_channels,exp_rate[3])
        
        self.med_4=torch.nn.Sequential(*[
            MedNext(8*in_channels,out_channels,exp_rate[3])
            for i in range(self.num_blocks[3])
        ])
        
        self.encoder_down4=MedDownSample(8*in_channels,out_channels,exp_rate[4])
        
        self.med_5=torch.nn.Sequential(*[
            MedNext(16*in_channels,out_channels,exp_rate[4])
            for i in range(self.num_blocks[4])
        ])
        
        self.decoder_up1=MedUpSample(16*in_channels,out_channels,exp_rate[4])
        
        self.med_6=torch.nn.Sequential(*[
            MedNext(8*in_channels,out_channels,exp_rate[5])
            for i in range(self.num_blocks[5])
        ])
        
        self.decoder_up2=MedUpSample(8*in_channels,out_channels,exp_rate[5])
            
        
        self.med_7=torch.nn.Sequential(*[
            MedNext(4*in_channels,out_channels,exp_rate[6])
            for i in range(self.num_blocks[6])
        ])
        
        self.decoder_up3=MedUpSample(4*in_channels,out_channels,exp_rate[6])
        
        self.med_8=torch.nn.Sequential(*[
            MedNext(2*in_channels,out_channels,exp_rate[7])
            for i in range(self.num_blocks[7])
        ])
        
        self.decoder_up4=MedUpSample(2*in_channels,out_channels,exp_rate[7])
            
        
        self.med_9=torch.nn.Sequential(*[
            MedNext(in_channels,out_channels,exp_rate[8])
            for i in range(self.num_blocks[8])
        ])
        
        self.out=torch.nn.ConvTranspose3d(in_channels,out_channels,1)
    
    def forward(self,x):
        #encoder section
        x1=self.semm(x)
        x2=self.med_1(x1)
        x3=self.encoder_down1(x2)
        x4=self.med_2(x3)
        x5=self.encoder_down2(x4)
        x6=self.med_3(x5)
        x7=self.encoder_down3(x6)
        x8=self.med_4(x7)
        x9=self.encoder_down4(x8)
        
        #bottle_next
        x10=self.med_5(x9)

        #decoder 
        x11=self.decoder_up1(x10)
        x12=self.med_6(x11+x8)
        x13=self.decoder_up2(x12)
        x14=self.med_7(x13+x6)
        x15=self.decoder_up3(x14)
        x16=self.med_8(x15+x4)
        x17=self.decoder_up4(x16)
        x18=self.med_9(x17+x2)
        x19=self.out(x18)
        x20=torch.nn.functional.softmax(x19)

        return x20