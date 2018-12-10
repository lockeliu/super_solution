import torch
from comm import load_data
from math import log10
from comm import comm
from comm.misc import progress_bar

class Trainer(object):
    def __init__(self, model_type, scale_list, train_data, val_data, model_path, train_batch_size = 64, input_img_size = 48, repeat = 10 , epoch = 100, lr = 0.0001, gpu_num=4):
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

    def build_model(self):
        trainset = load_data.MyDataSet( self.train_data, self.scale_list, 'train', self.train_batch_size, self.input_img_size, self.repeat )
        valset = load_data.MyDataSet( self.val_data, self.scale_list, 'test')

        self.training_loader = torch.utils.data.DataLoader( trainset , batch_size = self.train_batch_size , shuffle = True, num_workers=20 )
        self.testing_loader = torch.utils.data.DataLoader( valset , batch_size = 1, shuffle = True, num_workers=20 )

        net = comm.get_model(self.model_type, self.scale_list, self.model_path )
        print(net)

        self.net = net.cuda()
        self.model = torch.nn.DataParallel( self.net, self.gpu_list )
        self.train_criterion = torch.nn.L1Loss()
        self.test_criterion = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.lr )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)

    def save(self):
        self.net.savemodel()

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (scale, data, target) in enumerate(self.training_loader):
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            loss = self.train_criterion(self.model(data, scale), target)
            loss *= 1
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (scale, data, target) in enumerate(self.testing_loader):
                data, target = data.cuda(), target.cuda()
                prediction = self.model(data, scale)
                mse = self.test_criterion(prediction, target)
                psnr = 10 * log10(255 * 255 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.epoch + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            self.save()
