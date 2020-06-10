'''Author: Xingyi Yang
  Affiliation: UC San Diego
'''
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.dataset import *
from models.Unet import UNet
from utils.iou import IoU
from utils.Loss import *
from models.ResUnet import ResUNet
import argparse
def train(model,train_loader,optimizer,LOSS_FUNC,EPOCH,PRINT_INTERVAL, epoch, device):
    losses = []
    for i, batch in enumerate(tqdm(train_loader)):
        img, label = batch['img'].to(device), batch['label'].to(device)
        output = model(img)
        optimizer.zero_grad()

        loss = LOSS_FUNC(output, label)

        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if (i + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                       % (epoch + 1, EPOCH, i + 1, len(train_loader), loss.item()))
    return np.mean(losses)

def eval(model,val_loader,LOSS_FUNC, device):
    losses = []
    for i, batch in enumerate(val_loader):
        img, label = batch['img'].to(device), batch['label'].to(device)
        output = model(img)
        loss = LOSS_FUNC(output, label)
        losses.append(loss.item())
    return np.mean(losses)


# In[12]:
def main():
    parser = argparse.ArgumentParser(description='Image Classification.')
    parser.add_argument('--image-dir', type=str, default='../data/dataset_5_10/data/4_4_data_crop')
    parser.add_argument('--mask-dir', type=str, default='../data/dataset_5_10/data/med_seg_lungmask')

    parser.add_argument('--train-COVID', type=str,
                        default='../data/dataset_5_10/train_COVID_all_old.txt')
    parser.add_argument('--train-NonCOVID', type=str,
                        default='../data/dataset_5_10/train_NonCOVID_all_old_and_real.txt')
    parser.add_argument('--resume',type=str,default='') # ./checkpoint/ResUnet/best.pth.tar
    parser.add_argument('--val-COVID', type=str,
                        default='../data/dataset_5_10/val_COVID.txt')
    parser.add_argument('--val-NonCOVID', type=str,
                        default='../data/dataset_5_10/val_NonCOVID.txt')

    parser.add_argument('--test-COVID', type=str,
                        default='../data/dataset_5_10/test_COVID.txt')
    parser.add_argument('--test-NonCOVID', type=str,
                        default='../data/dataset_5_10/test_NonCOVID.txt')
    parser.add_argument('--start-epoch',type = int ,default=0, help='Start training epoch')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/ResUnet/')
    args = parser.parse_args()

    if os.path.exists(args.checkpoint) == False:
        os.makedirs(args.checkpoint)

    transform = transforms.Compose([
            RandomRescale(0.6,1.5),
            RandomCrop((320, 320)),
            RandomFlip(),
            RandomRotation(),
            RandomColor(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                      std=[0.5, 0.5, 0.5])
        ])
    test_transform = transforms.Compose([
            Resize((320, 400)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                      std=[0.5, 0.5, 0.5])

        ])
    train_dst = SegCOVICTDataset(image_dir=args.image_dir,mask_dir=args.mask_dir,
                                 covid_txt=args.train_COVID,non_covid_txt=args.train_NonCOVID,
                               transforms=transform)
    valid_dst = SegCOVICTDataset(image_dir=args.image_dir,mask_dir=args.mask_dir,
                                 covid_txt=args.val_COVID,non_covid_txt=args.val_NonCOVID,
                               transforms=test_transform)
    test_dst = SegCOVICTDataset(image_dir=args.image_dir, mask_dir=args.mask_dir,
                                 covid_txt=args.test_COVID, non_covid_txt=args.test_NonCOVID,
                                 transforms=test_transform)
    batch_size = 16
    print("Train set {}\nValidation set {}\nTest set {}".format(len(train_dst),len(valid_dst),len(test_dst)))

    train_loader = DataLoader(train_dst,batch_size=batch_size,num_workers=8,shuffle=True)
    val_loader = DataLoader(valid_dst,batch_size=2,num_workers=8)
    test_loader = DataLoader(test_dst,batch_size=2,num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet().to(device)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    LOSS_FUNC = nn.CrossEntropyLoss().to(device)
    PRINT_INTERVAL = 5
    EPOCH= 100
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        state_dict = checkpoint['state_dict']
        msg = model.load_state_dict(state_dict)
        print("==load model {}".format(args.resume))

    # In[14]:

    val_loss_epoch = []
    for epoch in range(args.start_epoch,EPOCH):

        model.train()
        train_loss = train(model, train_loader, optimizer, LOSS_FUNC, EPOCH, PRINT_INTERVAL, epoch, device)
        val_loss = eval(model, val_loader, LOSS_FUNC, device)
        val_loss_epoch.append(val_loss)
        lr_sheduler.step()
        tqdm.write('Epoch [%d/%d], Aveage Train Loss: %.4f, Aveage Valiation Loss: %.4f'
                   % (epoch + 1, EPOCH, train_loss, val_loss))

        if val_loss == np.min(val_loss_epoch):
            print('Model saved')
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.checkpoint,'best.pth.tar'))

def evaluate():
    parser = argparse.ArgumentParser(description='Image Classification.')
    parser.add_argument('--image-dir', type=str, default='../data/dataset_5_10/data/4_4_data_crop')
    parser.add_argument('--mask-dir', type=str, default='../data/dataset_5_10/data/med_seg_lungmask')

    parser.add_argument('--train-COVID', type=str,
                        default='../data/dataset_5_10/train_COVID_all_old.txt')
    parser.add_argument('--train-NonCOVID', type=str,
                        default='../data/dataset_5_10/train_NonCOVID_all_old_and_real.txt')
    parser.add_argument('--resume',type=str,default='./checkpoint/ResUnet/best.pth.tar') # ./checkpoint/ResUnet/best.pth.tar
    parser.add_argument('--val-COVID', type=str,
                        default='../data/dataset_5_10/val_COVID.txt')
    parser.add_argument('--val-NonCOVID', type=str,
                        default='../data/dataset_5_10/val_NonCOVID.txt')

    parser.add_argument('--test-COVID', type=str,
                        default='../data/dataset_5_10/test_COVID.txt')
    parser.add_argument('--test-NonCOVID', type=str,
                        default='../data/dataset_5_10/test_NonCOVID.txt')

    args = parser.parse_args()


    test_transform = transforms.Compose([
            Resize((320, 400)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                      std=[0.5, 0.5, 0.5])

        ])

    test_dst = SegCOVICTDataset(image_dir=args.image_dir, mask_dir=args.mask_dir,
                                 covid_txt=args.test_COVID, non_covid_txt=args.test_NonCOVID,
                                 transforms=test_transform)


    test_loader = DataLoader(test_dst,batch_size=2,num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet().to(device)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model).to(device)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        state_dict = checkpoint['state_dict']
        msg = model.load_state_dict(state_dict)
        print("==load model {}".format(args.resume))
    LOSS = DiceCELoss()
    # In[14]:
    num_classes = 2
    metric = IoU(num_classes)
    epoch_loss = 0.0
    for step, batch in enumerate(test_loader):
        # Get the inputs and labels
        img, label = batch['img'].to(device), batch['label'].to(device)
        with torch.no_grad():
            # Forward propagation
            outputs = model(img)

            # Loss computation
            loss = LOSS(outputs, label)

        # Keep track of loss for current epoch
        epoch_loss += loss.item()

        # Keep track of evaluation the metric
        metric.add(outputs.detach(), label.detach())
    print("Test Loss {}\tmIOU {}".format(epoch_loss/ len(test_loader), metric.value()))

if __name__ == '__main__':
    # main()
    evaluate()