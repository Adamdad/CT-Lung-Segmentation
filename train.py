'''Author: Xingyi Yang
  Affiliation: UC San Diego
'''
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.dataset import *
from models.Unet import UNet

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
    transform = transforms.Compose([
            RandomRescale(0.5,1.2),
            RandomCrop((224,224)),
            RandomColor(),
            RandomFlip(),
            RandomRotation(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                      std=[0.5, 0.5, 0.5])
        ])
    test_transform = transforms.Compose([
            Resize((224,224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                      std=[0.5, 0.5, 0.5])
        ])
    train_dst = SegCTDataset(txt='../Segmen/train.txt',
                               transforms=transform)
    valid_dst = SegCTDataset(txt='../Segmen/test.txt',
                               transforms=test_transform)
    batch_size = 16
    print("Train set {}\nValidation set {}\n".format(len(train_dst),len(valid_dst)))

    train_loader = DataLoader(train_dst,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(valid_dst,batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    LOSS_FUNC = nn.CrossEntropyLoss().to(device)
    PRINT_INTERVAL = 5
    EPOCH= 100


    # In[14]:

    val_loss_epoch = []
    for epoch in range(EPOCH):

        model.train()
        train_loss = train(model, train_loader, optimizer, LOSS_FUNC, EPOCH, PRINT_INTERVAL, epoch, device)
        val_loss = eval(model, test_loader, LOSS_FUNC, device)
        val_loss_epoch.append(val_loss)
        lr_sheduler.step(epoch)
        tqdm.write('Epoch [%d/%d], Aveage Train Loss: %.4f, Aveage Valiation Loss: %.4f'
                   % (epoch + 1, EPOCH, train_loss, val_loss))

        if val_loss == np.min(val_loss_epoch):
            print('Model saved')
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, '../Segmen/best.pth.tar')


if __name__ == '__main__':
    main()