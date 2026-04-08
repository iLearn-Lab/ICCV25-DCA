import random
import time
import math
import warnings
import argparse
import shutil
import os.path as osp
from typing import Tuple

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

import utils
from tllib.alignment.mcd import ImageClassifierHead, classifier_discrepancy
from tllib.alignment.mixup import mixup_p_data, mixup_data, mixup_criterion, mixup_focal_loss, MixUpSourceTarget, MixUpSourceTargetLinear
from tllib.alignment.bc_source_only import ImageClassifierHeadMode4
from tllib.alignment.bsp import SpectralDebiasingLoss, BSPLoss
from tllib.modules.my_loss import entropy
from tllib.utils.data import ForeverDataIterator, ClassAwareSampler, get_sampler
from tllib.utils.metric import accuracy, ConfusionMatrix, multi_label_auc, auc, binary_accuracy, binary_accuracy_original, multi_label_accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
from tllib.utils.my_loss import focalLoss
from tllib.self_training.pseudo_label import get_mask, get_hi_confidence_samples, get_pseudo_label_acc

from tllib.alignment.class_map import ClassMap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_map = ClassMap().class_map



def main(args:argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase, args.resume, args.resume_path)
    print(args.note, end="\n\n")
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = False
    cudnn.deterministic = True 


    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)


    train_source_dataset, train_target_dataset, target_val_dataset, target_test_dataset, source_val_dataset, source_test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform, sl=True, class_index=args.current_class)
    print(num_classes)
    print(args.class_names)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    
    target_val_loader = DataLoader(target_val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    target_test_loader = DataLoader(target_test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    source_val_loader = DataLoader(source_val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    source_test_loader = DataLoader(source_test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)


    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    
    # create model
    print("=> using model '{}'".format(args.arch))
    # backbone
    G = utils.get_model(args.arch, pretrain=not args.scratch).to(device)  # feature extractor
    pool_layer = nn.Identity() if args.no_pool else None
    F = ImageClassifierHeadMode4(G.out_features, args.bottleneck_dim, pool_layer, num_classes=1).to(device)
    sd_penalty = SpectralDebiasingLoss().to(device)

    # define loss function
    cls_loss_function = nn.BCEWithLogitsLoss(reduction="mean")


    # resume from the best checkpoint
    if args.phase != 'train':
        #checkpoint = torch.load(osp.join(logger.get_checkpoint_root(), args.model_path), map_location='cpu')
        checkpoint = torch.load(osp.join(args.model_path), map_location='cpu')
        G.load_state_dict(checkpoint['g'])
        F.load_state_dict(checkpoint['f'])


    if args.phase == 'test':
        if args.uncertain_test:
            uncertain_dataset, without_uncertain_test_dataset = utils.get_single_uncertain_dataset(args.data, args.root, args.target,  train_transform, val_transform, sl=True, class_index=args.current_class)
            uncertain_loader = DataLoader(uncertain_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
            without_uncertain_test_loader = DataLoader(without_uncertain_test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
            uncertain_validate(uncertain_loader, G, F, args, type="Target uncertain")
            target_valid_auc, loss= validate(without_uncertain_test_loader, G, F, args, type="Target without uncertain")
            print("Target without uncertain")
            print('test loss:{:3.1f}'.format(loss))
            print('auc:', end='\t')
            print(target_valid_auc)
 

        target_valid_auc, loss, count_list = validate(target_test_loader, G, F, args, type="Target")
        print("Target")
        print('test loss:{:3.1f}'.format(loss))
        print('auc:', end='\t')
        print(target_valid_auc)
        print(count_list)
       
        return
    
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(G,F.pool_layer).to(device)
        source_feature = collect_feature(source_test_loader, feature_extractor, device)
        print(source_feature.shape)
        target_feature = collect_feature(target_test_loader, feature_extractor, device)
        print(target_feature.shape)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return
    


    start_epoch = 0
    best_auc_current_class = 0.

    if args.pretrain:
        checkpoint = torch.load(args.pretrain_model_path)
        G.load_state_dict(checkpoint['g'])

    if args.decouple:   
        # dataloader
        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, sampler=get_sampler(train_source_dataset.targets),
                                     shuffle=False, num_workers=args.workers, drop_last=True)
        train_source_iter = ForeverDataIterator(train_source_loader)
     

    # start training
    valid_loss_list = []
    valid_auc_list = []
    train_loss_list = []
    train_st_loss_list = []
    train_sd_loss_list = []
    train_auc_list = []
  
    


    if args.resume == True:
        checkpoint = torch.load(logger.get_checkpoint_path('latest'))
        G.load_state_dict(checkpoint['g'])
        F.load_state_dict(checkpoint['f'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best_auc_current_class = checkpoint['best_auc']

    print('train {}'.format(class_map[args.current_class]))
    optimizer = SGD([{'params': G.parameters(), 'lr':0.1}, {'params': F.head.parameters(), 'lr':1.0}], lr=1,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    mixup_source_target = MixUpSourceTarget(alpha=args.lam_alpha, sup=args.sup)

    for epoch in range(start_epoch, args.epochs):
        print("lr:", lr_scheduler.get_last_lr())
        train_loss, train_auc, loss_st, loss_sd = train(train_source_iter, train_target_iter, 
                                            G, F, mixup_source_target, cls_loss_function,sd_penalty,
                                            optimizer, lr_scheduler, epoch, args)
       
        train_auc_list.extend(train_auc)
        train_loss_list.extend(train_loss)
        train_st_loss_list.extend(loss_st)
        train_sd_loss_list.extend(loss_sd)



        # source_valid_auc, source_valid_loss = validate(source_val_loader, G, F, args, type="Source")
        target_valid_auc, target_valid_loss = validate(target_val_loader, G, F, args, type="Target")
        auc_current_class = target_valid_auc
        valid_auc_list.append(target_valid_auc)
        valid_loss_list.append(target_valid_loss)

       
        torch.save({'g':G.state_dict(),
                    'f':F.state_dict(),
                    'head':F.head.state_dict(),
                    'epoch':epoch+1,
                    "best_auc":best_auc_current_class,
                    'class':args.current_class}, 
                    logger.get_checkpoint_path('latest'))
        if auc_current_class > best_auc_current_class:
            print("best model:"+ str(epoch))
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_auc_current_class = max(auc_current_class, best_auc_current_class)

        visualize(valid_loss_list, valid_auc_list, train_loss_list, train_auc_list, 
                  train_sd_loss_list, train_st_loss_list, 
                  path = logger.get_visualize_path(), class_name=class_map[args.current_class])

      
       

    # evaluate on test set
    G.load_state_dict(torch.load(logger.get_checkpoint_path('best'))['g'])
    F.load_state_dict(torch.load(logger.get_checkpoint_path('best'))['f'])
    source_test_auc, source_test_loss = validate(source_test_loader, G, F, args, type="Source")
    target_test_auc, target_test_loss = validate(target_test_loader, G, F, args, type="Target")


    print("Evaluate with the best model on target")
    print('target_auc   :{:3.1f}'.format(target_test_auc))
    print('target loss  :{:3.1f}'.format(target_test_loss))
    
    print("Evaluate with the best model on source")
    print('source_auc   :{:3.1f}'.format(source_test_auc))
    print('source loss  :{:3.1f}'.format(source_test_loss))
    
    print("Train {} Over".format(class_map[args.current_class]))
    logger.close()

   


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, 
          G: nn.Module, F: nn.Module, mixup_source_target: MixUpSourceTarget,
          cls_loss_function: nn.BCEWithLogitsLoss, sd_penalty: SpectralDebiasingLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    loss_s_meter = AverageMeter('Loss Source', ':6.2f')
    loss_st_meter = AverageMeter('Loss ST', ':6.2f')
    loss_sd_meter = AverageMeter('Loss SD', ':6.2f')
    auc_s_meter = AverageMeter('source '+class_map[args.current_class], ':3.1f')
    auc_t_meter = AverageMeter('target '+class_map[args.current_class], ':3.1f')
    mask_p_meter = AverageMeter('mask p num', ':3.1f')
    mask_n_meter = AverageMeter('mask n num', ':3.1f')
    pseudo_acc_p_meter = AverageMeter('pseudo acc p', ':3.1f')
    pseudo_acc_n_meter = AverageMeter('pseudo acc n', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, loss_s_meter, loss_st_meter, loss_sd_meter, auc_s_meter, auc_t_meter, mask_p_meter, mask_n_meter, pseudo_acc_p_meter, pseudo_acc_n_meter], 
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    G.train()
    F.train()
   

    end = time.time()
    loss_s_list = []
    loss_st_list = []
    loss_sd_list = []
    auc_t_list = []

    if args.iters_per_epoch == 0 :
        args.iters_per_epoch = train_source_iter.__len__()
    print_freq = math.ceil(args.iters_per_epoch / 10.)

    

    for i in range(args.iters_per_epoch):
        # source data
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device).float()
        # target data
        x_t, labels_t = next(train_target_iter)[:2]
        x_t = x_t.to(device)
        labels_t = labels_t.to(device).float()

        # measure data loading time
        data_time.update(time.time() - end)

        # source cls
        if args.source_mixup_mode == 0:
            g_s = G(x_s) # batchsize * 2048
            y_s, g_s = F(g_s, get_f=True)   # batchsize 
            cls_loss_s = cls_loss_function(y_s, labels_s)
            cls_auc_s = auc(y_s, labels_s) 
            if len(cls_auc_s)>0:
                auc_s_meter.update(cls_auc_s[-1], x_s.size(0))

        elif args.source_mixup_mode == 1:
            x_s, labels_s_a, labels_s_b, lam_s = mixup_data(x_s, labels_s, device, args.source_mixup_alpha)
            g_s = G(x_s) # batchsize * 2048
            y_s, g_s = F(g_s, get_f=True)   # batchsize 
            cls_loss_s = mixup_criterion(cls_loss_function, y_s, labels_s_a, labels_s_b, lam_s)
            cls_auc_a = auc(y_s, labels_s_a)
            cls_auc_b = auc(y_s, labels_s_b)
            if len(cls_auc_a)>0:
                auc_s_meter.update(lam_s * cls_auc_a[-1] + (1-lam_s) * cls_auc_b[-1], x_s.size(0))
        loss = cls_loss_s
        loss_s_meter.update(cls_loss_s.item(), x_s.size(0))


        # target detach cls
        g_t = G(x_t)
        y_t, g_t = F(g_t, get_f=True)
        y_t = y_t.detach() # batchsize
      
        if args.sd:
            mask_f_p, mask_f_n = get_mask(args.f_hi_threshold, args.f_lo_threshold, y_t)
            sd_loss = sd_penalty(g_s, 
                                torch.cat((g_t[mask_f_n==1], g_s[labels_s==0]),dim=0),
                                torch.cat((g_t[mask_f_p==1], g_s[labels_s==1]),dim=0),)


        
        loss += args.trade_off_sd * sd_loss
        loss_sd_meter.update(sd_loss.item(), x_s.size(0))

        cls_auc_t = auc(y_t, labels_t)
        if len(cls_auc_t)>0:
            auc_t_meter.update(cls_auc_t[-1], x_t.size(0))
        mask_p, mask_n = get_mask(args.hi_threshold,args.lo_threshold, y_t)
        mask_p_meter.update(mask_p.sum(dim=0).item(), 1)
        mask_n_meter.update(mask_n.sum(dim=0).item(), 1)
        acc_p, acc_n = get_pseudo_label_acc(args.hi_threshold,args.lo_threshold, y_t, labels_t)
        if torch.isnan(acc_p) == False:
            pseudo_acc_p_meter.update(acc_p.item(), mask_p.sum(dim=0).item())
        if torch.isnan(acc_n) == False:
            pseudo_acc_n_meter.update(acc_n.item(), mask_n.sum(dim=0).item())

        if args.no_repeat_target:
            x_t, pseudo_labels_t = get_hi_confidence_samples(args.hi_threshold,args.lo_threshold,x_t,y_t,device)
        else:
            if args.target_class_balance:
                x_t, pseudo_labels_t = get_hi_confidence_samples(args.hi_threshold,args.lo_threshold,x_t,y_t,device,x_t.size(0),class_balance=True)
            else:
                x_t, pseudo_labels_t = get_hi_confidence_samples(args.hi_threshold,args.lo_threshold,x_t,y_t,device,x_t.size(0),class_balance=False)
        if x_t == None:
            pass 
        else:
            # source target mixup
            mixed_x_st, labels_st_a, labels_st_b, lam_st = mixup_source_target(x_s, x_t, labels_s, pseudo_labels_t, device)
            
            # st mixup cls
            g_st = G(mixed_x_st)
            y_st = F(g_st)
            cls_loss_st = mixup_criterion(cls_loss_function, y_st, labels_st_a, labels_st_b, lam_st)


            loss += args.trade_off_st * cls_loss_st
            loss_st_meter.update(cls_loss_st.item(), x_s.size(0))

        
            

        # backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            progress.display(i)
            loss_s_list.append(loss_s_meter.val)
            auc_t_list.append(auc_t_meter.val)
            loss_st_list.append(loss_st_meter.val)
            loss_sd_list.append(loss_sd_meter.val)
            # print(mixup_source_target.coeff)

    return loss_s_list, auc_t_list, loss_st_list, loss_sd_list


def validate(val_loader: DataLoader, G: nn.Module, F: ImageClassifierHead, args: argparse.Namespace, type):
    larger_list = []
    print(' Validate on the {}'.format(type))
    batch_num = len(val_loader)
    print_freq = math.ceil(batch_num / 5)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.2f')
    auc_meter = AverageMeter(type + ' ' + class_map[args.current_class], ':3.1f')
    cls_acc_meter = AverageMeter(class_map[args.current_class], ':3.1f')
    cls_n_acc_meter = AverageMeter(class_map[args.current_class], ':3.1f')
    cls_p_acc_meter = AverageMeter(class_map[args.current_class], ':3.1f')
    
    p_norm_meter = AverageMeter(class_map[args.current_class]+' '+'p_norm', ':3.1f')
    n_norm_meter = AverageMeter(class_map[args.current_class]+' '+'n_norm', ':3.1f')


    

    
    p_score_meter = AverageMeter(class_map[args.current_class]+' '+'p_score', ':3.1f')
    n_score_meter = AverageMeter(class_map[args.current_class]+' '+'n_score', ':3.1f')

    progress = ProgressMeter(
        batch_num,
        [batch_time, losses, auc_meter, p_norm_meter, n_norm_meter],
        prefix="{} Test: ".format(type))
      

    cls_loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    # switch to evaluate mode
    G.eval()
    F.eval()
    # old_ece_list = []
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device).float()

            # compute output
            g = G(images)
            y = F(g)
            cls_loss = cls_loss_function(y, target)

            norm = torch.norm(g, p=1,dim=1)
            p_norm_avg = torch.mean(norm[target == 1])
            p_norm_meter.update(p_norm_avg.item(), target[target == 1].shape[0])
            n_norm_avg = torch.mean(norm[target == 0])
            n_norm_meter.update(n_norm_avg.item(), target[target == 0].shape[0])

            y_sigmoid = torch.sigmoid(y)
            pred = (y_sigmoid >= 0.5).float().t().view(-1)
            p_score_avg = torch.mean(y_sigmoid[target == 1])
            p_score_meter.update(p_score_avg.item(), y_sigmoid[target == 1].shape[0])
            n_score_avg = torch.mean(y_sigmoid[target == 0])
            n_score_meter.update(n_score_avg.item(), y_sigmoid[target == 0].shape[0])
            mask_p = (target==1).float()
           

            cls_acc = binary_accuracy_original(y, target, 0.5)
            cls_acc_meter.update(cls_acc,images.size(0))
            p_acc,n_acc = binary_accuracy(y, target)
            if torch.isnan(p_acc) == False:
                cls_p_acc_meter.update(p_acc,images.size(0))
            if torch.isnan(n_acc) == False:
                cls_n_acc_meter.update(n_acc,images.size(0))
            cls_auc = auc(y, target)
            if len(cls_auc)>0:
                auc_meter.update(cls_auc[-1],images.size(0))

          
           
            losses.update(cls_loss.item(), images.size(0))


            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                progress.display(i)
            
        print(' * Average AUC {top1.avg:.3f}'.format(top1=auc_meter))


    res = auc_meter.avg
    print("正例准确率 {:.3f}".format(cls_p_acc_meter.avg))
    print("负例准确率 {:.3f}".format(cls_n_acc_meter.avg))
    print("平均准确率 {:.3f}".format(cls_acc_meter.avg))
    print("正例平均得分 {:.3f}".format(p_score_meter.avg))
    print("负例平均得分 {:.3f}".format(n_score_meter.avg))
    print("正例平均范数 {:.3f}".format(p_norm_meter.avg))
    print("负例平均范数 {:.3f}".format(n_norm_meter.avg))


    # ece_list = np.mean(np.array(larger_list), axis=0)
    # row_max = np.max(np.array(larger_list), axis=0)
    # row_min = np.min(np.array(larger_list), axis=0)
    # mean_arr = np.add(row_max, row_min) / 2
    # print(row_max)
    # print(row_min)
    # print(mean_arr)
    return res, losses.avg


# 可略过
def uncertain_validate(val_loader: DataLoader, G: nn.Module, F: ImageClassifierHead, args: argparse.Namespace, type):
    print(' Validate on the {}'.format(type))
    batch_num = len(val_loader)
    print_freq = math.ceil(batch_num / 5)
    batch_time = AverageMeter('Time', ':6.3f')
    uncertain_score_meter = AverageMeter(class_map[args.current_class]+' '+'uncertain_score', ':6.3f')
    uncertain_score_var_meter = AverageMeter(class_map[args.current_class]+' '+'uncertain_score_var', ':6.3f')
    progress = ProgressMeter(
        batch_num,
        [batch_time, uncertain_score_meter],
        prefix="{} Test: ".format(type))


    # switch to evaluate mode
    G.eval()
    F.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device).float()

            # compute output
            g = G(images)
            y = F(g)
            

            y_sigmoid = torch.sigmoid(y)
            uncertain_score_avg = torch.mean(y_sigmoid[target == -1])
            uncertain_score_var = torch.var(y_sigmoid[target == -1])
            uncertain_score_meter.update(uncertain_score_avg.item(), y_sigmoid[target == -1].shape[0])
            uncertain_score_var_meter.update(uncertain_score_var.item(), y_sigmoid[target == -1].shape[0])

            

            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                progress.display(i)
            
       
    print("不确定例平均得分 {:.3f}".format(uncertain_score_meter.avg))
    print("不确定例平均得分方差 {:.3f}".format(uncertain_score_var_meter.avg))



def visualize(valid_loss_list, valid_auc_list, train_loss_list, train_auc_list, train_sd_loss_list, train_st_loss_list, path, class_name):
    plt.figure(figsize=(200, 80), dpi=100)
    plt.rc('font', size=80)

    plt.subplot(2, 3, 1)
    try:
        train_loss_lines.remove(train_loss_lines[0]) 
    except Exception:
        pass
    train_loss_lines = plt.plot(list(range(len(train_loss_list))), train_loss_list, 'green', lw=1) 
    plt.title("train_loss")
    plt.xlabel("iter")
    plt.ylabel("train_loss")
    plt.legend(labels=['train_loss'])

    plt.subplot(2, 3, 2)
    try:
        train_auc_line.remove(train_auc_line[0]) 
    except Exception:
        pass
    train_auc_line = plt.plot(list(range(len(train_auc_list))), train_auc_list, 'r', lw=1)  
    plt.title("train_auc")
    plt.xlabel("iter")
    plt.ylabel("train_auc")
    plt.legend(labels=[class_name])

    plt.subplot(2, 3, 3)
    try:
        valid_loss_lines.remove(valid_loss_lines[0]) 
    except Exception:
        pass
    valid_loss_lines = plt.plot(list(range(len(valid_loss_list))), valid_loss_list, 'b', lw=1)
    plt.title("valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("valid_loss")
    plt.legend(labels=['valid_loss'])

    plt.subplot(2, 3, 4)
    try:
        valid_auc_line.remove(valid_auc_line[0])  
    except Exception:
        pass
    valid_auc_line = plt.plot(list(range(len(valid_auc_list))), valid_auc_list, 'r', lw=2) 
    plt.title("valid_auc")
    plt.xlabel("epoch")
    plt.ylabel("valid_auc")
    plt.legend(labels=[class_name])

    plt.subplot(2, 3, 5)
    try:
        train_svd_loss_line.remove(train_svd_loss_line[0])  
    except Exception:
        pass
    train_svd_loss_line = plt.plot(list(range(len(train_sd_loss_list))), train_sd_loss_list, 'b', lw=2) 
    plt.title("sd loss")
    plt.xlabel("iter")
    plt.ylabel("sd loss")
    plt.legend(labels=['sd loss'])

    plt.subplot(2, 3, 6)
    try:
        train_st_loss_line.remove(train_st_loss_line[0])  
    except Exception:
        pass
    train_st_loss_line = plt.plot(list(range(len(train_st_loss_list))), train_st_loss_list, 'green', lw=2) 
    plt.title("train st loss")
    plt.xlabel("iter")
    plt.ylabel("train st loss")
    plt.legend(labels=['train st loss'])

    plt.show()
    plt.savefig(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised_Adversarial_Domain_Adaptation_for_Multi-Label_Classification_of_Chest_X-Ray based on multi_label DANN ')


    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='medical_images', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: medical_images)')
    parser.add_argument('-s', '--source', default="NIH_CXR14", help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', default="Open-i", help='target domain(s)', nargs='+')


    parser.add_argument('--note',type = str, default = '',
                        help = 'note about this train')
   
    parser.add_argument('--preresized', action='store_true', help = 'ues pre-resized 256*256 images')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')




    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')



    # training parameters
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--test-batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size (default: 512)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')



    parser.add_argument('-i', '--iters-per-epoch', default=0, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='multi_label_dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis', 'analysis2'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    
    parser.add_argument('--resume', action='store_true', help = 'ues pre-resized 256*256 images')
    parser.add_argument("--resume-path", type=str, default = '',
                        help="the last train logg path")
    parser.add_argument('--model-path',type = str, default = '',
                        help = 'when test or analysis ,  the path where the choosen model')
    
    parser.add_argument('--focal-gamma', default=1., type=float, help='parameter for focal loss')

    parser.add_argument('--without-normal', action='store_true', help='Whether the classes contain no finding')

    parser.add_argument('--trade-off-entropy', default=0.1, type=float,
                        help='the trade-off hyper-parameter for entropy loss')
    
    parser.add_argument('--classifier-mode', default=1, type=int, choices=[1, 2, 3, 4])
    
    parser.add_argument('--train-mode', default='train_g', type=str, choices=['train_g', 'train_f'])
    parser.add_argument("--pretrain",action='store_true', help = 'whether use pretrained model')
    parser.add_argument('--pretrain-model-path', default='', type=str)

    parser.add_argument('--source-mixup-alpha', default=1., type=float, help='Beta(source-mixup-alpha, source-mixup-alpha)')
    parser.add_argument("--source-mixup-mode", type=int, default=0, choices=[0, 1, 2, 3, 4])

    parser.add_argument('-c', '--current-class', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 275, 25, 225, 2125, 20625])
    parser.add_argument('--hi-threshold', default=0.6, type=float)
    parser.add_argument('--lo-threshold', default=0.4, type=float)

    
    parser.add_argument('--lam-alpha', default=1., type=float)
    parser.add_argument('--trade-off-st', default=1., type=float,
                        help='the trade-off-st hyper-parameter for cls st loss')
    

    parser.add_argument("--decouple", action='store_true', help = 'whether use pretrained model')

    parser.add_argument("--no-repeat-target",action='store_true', help = 'whether use pretrained model')

    parser.add_argument('--uncertain-test', action='store_true')

    parser.add_argument('--target-class-balance', action='store_true')


    # spectral debiasing
    parser.add_argument('--trade-off-sd', default=2e-4, type=float,
                        help='the trade-off hyper-parameter for sd loss')
    parser.add_argument('--sd', action='store_true')
    
    parser.add_argument('--f-hi-threshold', default=0.7, type=float)
    parser.add_argument('--f-lo-threshold', default=0.1, type=float)
    parser.add_argument('--sup', default=10, type=float)

    parser.add_argument('--ablation', default=None, type=str)

    args = parser.parse_args()
    main(args)
