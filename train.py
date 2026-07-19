import os
import sys
from omegaconf import OmegaConf
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from models import *
from datasets import *
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))
from common import losses, optimizers
from common.utils import *
from utils import *
from sklearn.metrics import roc_auc_score
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import OneCycleLR
from metrics_plotter import MetricsPlotter
args = get_params()
setup(args)
init_exam_dir(args)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1" #禁用版本检查


def main():
    global device
    plotter = MetricsPlotter(args.exam_dir)
    args.local_rank = 0
    logger = get_logger(str(args.local_rank), console=args.local_rank==0, 
        log_path=os.path.join(args.exam_dir, f'train_{args.local_rank}.log'))
    torch.backends.cudnn.benchmark = True
    train_dataloader = get_dataloader(args, 'train')
    test_dataloader = get_dataloader(args, 'test')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = eval(args.model.name)(**args.model.params)
    model = model.to(device)

    optimizer = optimizers.__dict__[args.optimizer.name](model.parameters(), **args.optimizer.params)
    criterion = losses.__dict__[args.loss.name](
        **(args.loss.params if getattr(args.loss, "params", None) else {})
    ).to(device)
    
    global_step = 1
    start_epoch = 1
    if args.model.resume:
        logger.info(f'resume from {args.model.resume}')
        checkpoint = torch.load(args.model.resume, map_location='cpu')
        if 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
            if (not getattr(args.model, 'only_resume_model', False)):
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
        else:
            sd = checkpoint

        not_resume_layer_names = args.model.not_resume_layer_names
        if not_resume_layer_names:
            for name in not_resume_layer_names:
                sd.pop(name)
                logger.info(f'Not loading layer {name}')
        model.load_state_dict(sd)

    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
        delta=0,
        path=os.path.join(args.exam_dir, 'checkpoint.pt')
    )
    for epoch in range(start_epoch, args.train.max_epoches):
        train_acc, train_loss=train(train_dataloader, model, criterion, optimizer, epoch, global_step, args, logger)
        global_step += len(train_dataloader)
        test_metrics = test(test_dataloader, model, criterion, optimizer, epoch, global_step, args, logger)
        early_stopping(test_metrics['test_loss'], model)
        plotter.update(
            train_loss=train_loss,
            val_loss=test_metrics['test_loss'],
            train_acc=train_acc,
            val_acc=test_metrics['test_acc'],
            val_auc=test_metrics['test_auc'],
        )
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    plotter.plot_all()

def train(dataloader, model, criterion, optimizer, epoch, global_step, args, logger):
    epoch_size = len(dataloader)
    model.set_segment(args.train.dataset.params.num_segments)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, data_time, batch_time])

    model.train()
    end = time.time()
    for idx, datas in enumerate(dataloader):
        optimizer.zero_grad()
        data_time.update(time.time() - end)

        # get input data from dataloader
        images, labels, video_paths, segment_indices,max_idxspic = datas
        images = images.to(device)
        labels = labels.to(device)
        max_idxspic = max_idxspic.to(device)
        combined_input = torch.cat((images, max_idxspic), dim=1)
        outputs = model(combined_input)

        # tune learning rate
        cur_lr = lr_tuner(args.optimizer.params.lr, optimizer, epoch_size, args.scheduler,
            global_step, args.train.use_warmup, args.train.warmup_epochs)

        loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)

        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))

        if (idx + 1) % args.train.print_info_step_freq == 0:
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)} lr: {cur_lr:.7f}')
        
        global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()
    return acces.avg, losses.avg

def test(dataloader, model, criterion, optimizer, epoch, global_step, args, logger):
    model.set_segment(args.test.dataset.params.num_segments)

    model.eval()
    y_outputs, y_labels = [], []
    loss_t = 0.
    torch.no_grad()
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(dataloader)):
            images, labels, video_paths, segment_indices,max_idxspic = datas
            images = images.to(device)
            labels = labels.to(device)
            max_idxspic = max_idxspic.to(device)
            combined_input = torch.cat((images, max_idxspic), dim=1)
            outputs = model(combined_input)
            loss = criterion(outputs, labels)
            loss_t += loss * labels.size(0)

            y_outputs.extend(outputs)
            y_labels.extend(labels)

    gather_y_outputs = gather_tensor(y_outputs, dist_=False, to_numpy=False)
    gather_y_labels  = gather_tensor(y_labels, dist_=False, to_numpy=False)
    _, y_outputs = torch.max(gather_y_outputs, dim=1)
    y_probs = F.softmax(gather_y_outputs, dim=1)[:, 1]
    auc = roc_auc_score(gather_y_labels.cpu().numpy(), y_probs.cpu().numpy())
    epoch_data = {
        'epoch': epoch,
        'probs': y_probs,
        'labels': gather_y_labels,
        'auc': auc
    }

    save_path = os.path.join(args.exam_dir, f'auc_data_epoch_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(epoch_data, f)

    acc, real_acc, fake_acc, _, _ = compute_metrics(gather_y_outputs, gather_y_labels)
    weight_acc = 0.
    if real_acc and fake_acc:
        weight_acc = 2 / (1 / real_acc + 1 / fake_acc)

    loss = (loss_t / len(dataloader.dataset))

    lr = optimizer.param_groups[0]['lr']
    logger.info(
        '[TEST] EPOCH-{} Step-{} ACC: {:.4f} AUC: {:.4f} RealACC: {:.4f} FakeACC: {:.4f} Loss: {:.5f} lr: {:.7f}  '.format(
            epoch, global_step, acc, auc, real_acc, fake_acc, loss, lr
        )
    )
    if args.local_rank == 0:
        test_metrics = {
            'test_acc': acc,
            'test_weight_acc': weight_acc,
            'test_real_acc': real_acc,
            'test_fake_acc': fake_acc,
            'test_loss': loss,
            'test_auc':auc,
            'lr': lr,
            "epoch": epoch
        }

        checkpoint = OrderedDict()
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['global_step'] = global_step
        checkpoint['metrics'] = test_metrics
        checkpoint['args'] = args

        checkpoint_save_name = \
            "Epoch-{}-Step-{}-ACC-{:.4f}-AUC-{:.4f}-RealACC-{:.4f}-FakeACC-{:.4f}-Loss-{:.5f}-LR-{:.6g}.tar".format(
                epoch, global_step, acc, auc, real_acc, fake_acc, loss, lr
            )
        checkpoint_save_dir = os.path.join(
            os.path.join(args.exam_dir, 'ckpt'), 
            checkpoint_save_name
        )
        torch.save(checkpoint, checkpoint_save_dir)
    return {
        'test_acc': acc,
        'test_weight_acc': weight_acc,
        'test_real_acc': real_acc,
        'test_fake_acc': fake_acc,
        'test_loss': loss.cpu(),
        'test_auc': auc,
        'lr': lr,
        "epoch": epoch
    }

if __name__ == '__main__':
    main()
