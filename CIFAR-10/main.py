import time
import torch
import Models
import torchvision
import torchvision.transforms as transforms

from torch import nn
from torch.utils.data import DataLoader, Subset
from utils import AverageMeter, KDLoss


def train(model, teacher, train_dataloader, optimizer, criterion, kd_loss, args, epoch):
    train_loss = AverageMeter()
    train_error = AverageMeter()

    Cls_loss = AverageMeter()
    Div_loss = AverageMeter()

    # Model on train mode
    model.train()
    teacher.eval()
    step_per_epoch = len(train_dataloader)
    
    for step, (inputs, labels) in enumerate(train_dataloader):
        start = time.time()
        inputs, labels = inputs.cuda(), labels.cuda()
        s_logits = model(inputs)

        with torch.no_grad():
            t_logits = teacher(inputs)

        # cls loss
        cls_loss = criterion(s_logits, labels) * args['cls_loss_factor']
        # KD loss
        div_loss = kd_loss(s_logits, t_logits) * min(1.0, epoch/args['warm_up'])
        # total loss
        loss = cls_loss + div_loss

        # measure accuracy and record loss
        batch_size = inputs.size(0)
        _, pred = s_logits.data.cpu().topk(1, dim=1)
        train_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        train_loss.update(loss.item(), batch_size)

        Cls_loss.update(cls_loss.item(), batch_size)
        Div_loss.update(div_loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - kd_loss: {:.3f} - cls_loss: {:.3f} - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time() - start), div_loss.item(), cls_loss.item(), train_loss.val, 1-train_error.val)

        print(s1 + s2, end='', flush=True)

    print()
    return Div_loss.avg, Cls_loss.avg, train_loss.avg, train_error.avg


def test(model, test_dataloader, criterion):
    test_loss = AverageMeter()
    test_error = AverageMeter()
    inf_time = AverageMeter()

    # Model on eval mode
    model.eval()

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute logits
            start = time.time()
            logits = model(images)
            _, pred = logits.data.cpu().topk(1, dim=1)
            end = time.time()

            loss = criterion(logits, labels)

            # measure accuracy and record loss
            batch_size = images.size(0)
            test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            test_loss.update(loss.item(), batch_size)
            inf_time.update(end - start, 1)

    return test_loss.avg, test_error.avg, inf_time.avg


def epoch_loop(model, teacher, train_set, test_set, args):
    # data loaders
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, pin_memory=True, num_workers=args['workers'])
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, pin_memory=True, num_workers=args['workers'])
    
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.DataParallel(model)
    model.to(device)
    teacher = nn.DataParallel(teacher)
    teacher.to(device)

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    kd_loss = KDLoss(kl_loss_factor=args['kd_loss_factor'], T=args['t']).to(device)

    # optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'], nesterov=True)
   
    start_epoch = 0

    # Training model
    for epoch in range(start_epoch, args['epochs']):
        print("Epoch {}/{}".format(epoch + 1, args['epochs']))

        _, _, train_epoch_loss, train_error = train(model=model, teacher=teacher, train_dataloader=train_loader, optimizer=optimizer, criterion=criterion, kd_loss=kd_loss, args=args, epoch=epoch)
        
        # Testing model
        test_epoch_loss, test_error, inf_time = test(model=model, test_dataloader=test_loader, criterion=criterion)
        
        s = "Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, Inf Time: {:.5f}".format(train_epoch_loss, 1-train_error, test_epoch_loss, 1-test_error, inf_time)
        print(s)
        

if __name__ == "__main__":
    args = {'batch_size': 100, 'workers': 4, 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4, 'epochs': 10, 't': 4.0, 'kd_loss_factor': 1.0, 'cls_loss_factor': 1.0, 'warm_up': 10, 'device': 'cuda', 'dataset': 'cifar10'}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root="Datasets", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="Datasets", train=False, download=True, transform=transform)

    # Reduced train set
    reduced_indices = []
    for i in range(10):
        for j in range(i * 5000, (i+1) * 5000 - 4500):
            reduced_indices.append(i)
    train_set = Subset(train_set, reduced_indices) # comment this out to use the full train set

    model = Models.resnet18(device=args['device'])
    teacher = Models.resnet50(device=args['device'], pretrained=True)

    for param in teacher.parameters():
        param.requires_grad = False

    epoch_loop(model=model, teacher=teacher, train_set=train_set, test_set=test_set, args=args)
