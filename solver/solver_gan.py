import torch
from comm import load_data
from math import log10
from model import EDSR
from model import MDSR
from model import NewNet
from model import SRGAN
from comm.misc import progress_bar


class Trainer(object):
    def __init__(self, model_type, scale_list, train_data, val_data, model_path, d_model_path, train_batch_size = 64, input_img_size = 48, repeat = 10 , epoch = 100, lr = 0.0001, gpu_num=4):
        self.scale_list = scale_list
        self.train_batch_size = train_batch_size
        self.input_img_size = input_img_size
        self.repeat = repeat
        self.train_data = train_data
        self.val_data = val_data
        self.model_type = model_type
        self.epoch = epoch
        self.lr = lr
        self.gpu_list = [ gpu_id for gpu_id in range( gpu_num ) ]
        self.model_path = model_path
        self.d_model_path = d_model_path

    def build_model(self):
        trainset = load_data.MyDataSet( self.train_data, self.scale_list, 'train', self.train_batch_size, self.input_img_size, self.repeat )
        valset = load_data.MyDataSet( self.val_data, self.scale_list, 'test')

        self.training_loader = torch.utils.data.DataLoader( trainset , batch_size = self.train_batch_size , shuffle = True, num_workers=20 )
        self.testing_loader = torch.utils.data.DataLoader( valset , batch_size = 1, shuffle = True, num_workers=20 )

        netG = self.get_model() 
        netD = SRGAN.Discriminator(self.d_model_path)
        print(netG)
        print(netD)

        self.netG = netG.cuda()
        self.modelG = torch.nn.DataParallel( self.netG, self.gpu_list )

        self.netD = netD.cuda()
        self.modelD = torch.nn.DataParallel( self.netD, self.gpu_list )

        self.criterionG = torch.nn.L1Loss()
        self.criterionD = torch.nn.BCELoss()

        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.optimizerD = torch.optim.SGD(self.netD.parameters(), lr=0.0001 / 100, momentum=0.9, nesterov=True)

        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, step_size=15, gamma=0.5)
        self.schedulerD = torch.optim.lr_scheduler.StepLR(self.optimizerD, step_size=15, gamma=0.5)

        self.test_criterion = torch.nn.MSELoss()

    def get_model(self):
        self.model_type = self.model_type.lower()
        if self.model_type == 'edsr':
            return EDSR.EDSR(self.scale_list, self.model_path ) 
        elif self.model_type == 'mdsr':
            return MDSR.MDSR( self.scale_list, self.model_path )
        elif self.model_type == 'newnet':
            return NewNet.NewNet( self.scale_list, self.model_path )
        else:
            print("no this model_type " + self.model_type)
            exit(-1)

    def save(self):
        self.netG.savemodel()
        self.netD.savemodel()

    def pretrain(self):
        self.modelG.train()
        train_loss = 0
        for batch_num, (scale, data, target) in enumerate(self.training_loader):
            data, target = data.to('cuda'), target.to('cuda')
            self.optimizerG.zero_grad()
            loss = self.criterionG(self.modelG(data, scale), target)
            loss *= 1
            train_loss += loss.item()
            loss.backward()
            self.optimizerG.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def train( self ):
        self.modelG.train()
        self.modelD.train()
        g_train_loss = 0
        d_train_loss = 0
        for batch_num, (scale, data, target) in enumerate(self.training_loader):
            real_label = torch.ones(data.size(0), 1).cuda()
            fake_label = torch.zeros(data.size(0), 1).cuda()
            data, target = data.to('cuda'), target.to('cuda')
            #train Discriminator
            self.optimizerD.zero_grad()
            d_real = self.modelD(target)
            d_real_loss = self.criterionD(d_real, real_label)

            d_fake = self.modelD(self.modelG(data,scale))
            d_fake_loss = self.criterionD(d_fake, fake_label)
            d_total = d_real_loss + d_fake_loss
            d_train_loss += d_total.item()
            d_total.backward()
            self.optimizerD.step()

            # train Generator
            self.optimizerG.zero_grad()
            g_real = self.modelG(data, scale)
            g_fake = self.modelD(g_real)
            gan_loss = self.criterionD(g_fake, real_label)
            mse_loss = self.criterionG(g_real, target)

            g_total = mse_loss + 1e-3 * gan_loss
            g_train_loss += g_total.item()
            g_total.backward()
            self.optimizerG.step()
            progress_bar(batch_num, len(self.training_loader), 'G_Loss: %.4f | D_Loss: %.4f' % (g_train_loss / (batch_num + 1), d_train_loss / (batch_num + 1)))

        print("    Average G_Loss: {:.4f}".format(g_train_loss / len(self.training_loader)))


    def test(self):
        self.modelG.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (scale, data, target) in enumerate(self.testing_loader):
                data, target = data.to('cuda'), target.to('cuda')
                prediction = self.modelG(data, scale)
                mse = self.test_criterion(prediction, target)
                psnr = 10 * log10(255 * 255 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))


    def run(self):
        self.build_model()
        for epoch in range( 1, 1 ):
            self.pretrain()
        self.test()
        for epoch in range(1, self.epoch + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.schedulerG.step(epoch)
            self.schedulerD.step(epoch)
            self.save()

