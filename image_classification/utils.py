"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time
import math
from PIL import Image

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, binary_accuracy, ConfusionMatrix, multi_label_accuracy,multi_label_auc
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits'] + ["medical_images"]


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None, without_normal=False, sl=False, class_index=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        target_val_dataset = target_test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        source_val_dataset = source_test_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name == "medical_images":
        if sl:
            train_source_dataset = datasets.__dict__['MedicalImages'](root, task=source[0], class_index=class_index, split='train',
                                                                      download=True, transform=train_source_transform)
            train_target_dataset = datasets.__dict__['MedicalImages'](root, task=target[0], class_index=class_index, split='train',
                                                                      download=True,transform=train_target_transform)
            source_val_dataset = source_test_dataset = datasets.__dict__['MedicalImages'](root, task=source[0], class_index=class_index, split='test',
                                                                                           download=True, transform=val_transform)
            target_val_dataset = target_test_dataset = datasets.__dict__['MedicalImages'](root, task=target[0], class_index=class_index, split='test',
                                                                                           download=True, transform=val_transform)
            class_names = datasets.MedicalImages.get_classes()
            num_classes = len(class_names)
        else:
            if without_normal:
                train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='train_without_normal', download=True,
                                                                transform=train_source_transform, without_normal=without_normal)
                train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='train_without_normal', download=True,
                                                                transform=train_target_transform, without_normal=without_normal)
                target_val_dataset = target_test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test_without_normal',
                                                                    download=True, transform=val_transform, without_normal=without_normal)
                source_val_dataset = source_test_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='test_without_normal',
                                                                    download=True, transform=val_transform, without_normal=without_normal)
            else:
                train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                                transform=train_source_transform)
                train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                                transform=train_target_transform)
                target_val_dataset = target_test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                    download=True, transform=val_transform)
                source_val_dataset = source_test_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='test',
                                                                    download=True, transform=val_transform)
            class_names = datasets.NIH_CXR14.get_classes(without_normal,)
            num_classes = len(class_names)

    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        target_val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))
        if dataset_name == 'DomainNet':
            target_test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            target_test_dataset = target_val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, target_val_dataset, target_test_dataset, source_val_dataset, source_test_dataset, num_classes, class_names





def get_single_dataset(dataset_name, root, source, train_source_transform, val_transform, preresized=False, mode='RGB',without_normal=False, sl=False, class_index=None):
    if dataset_name == "medical_images":
        if sl:
            train_source_dataset = datasets.__dict__['MedicalImages'](root, task=source[0], class_index=class_index, split='train',
                                                                      download=True, transform=train_source_transform)
            source_val_dataset = source_test_dataset = datasets.__dict__['MedicalImages'](root, task=source[0], class_index=class_index, split='test',
                                                                                          download=True, transform=val_transform)
            class_names = datasets.MedicalImages.get_classes()
            num_classes = len(class_names)
        else:    
            if preresized:
                train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='train_resized', download=True,
                                                                transform=train_source_transform, mode=mode)
                source_val_dataset = source_test_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='test_resized',
                                                                    download=True, transform=val_transform, mode=mode)
            elif without_normal:
                train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='train_without_normal', download=True,
                                                                transform=train_source_transform, mode=mode,without_normal=without_normal)
                source_val_dataset = source_test_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='test_without_normal',
                                                                    download=True, transform=val_transform, mode=mode,without_normal=without_normal)
            else:
                train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='train', download=True,
                                                                transform=train_source_transform, mode=mode)
                source_val_dataset = source_test_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), split='test',
                                                                    download=True, transform=val_transform, mode=mode)
            class_names = datasets.NIH_CXR14.get_classes(without_normal)
            num_classes = len(class_names)

    return train_source_dataset, source_val_dataset, source_test_dataset, num_classes, class_names

def get_single_uncertain_dataset(dataset_name, root, source, train_source_transform, val_transform, preresized=False, mode='RGB',without_normal=False, sl=False, class_index=None):
    if dataset_name == "medical_images":
        if sl:
            uncertain_dataset = datasets.__dict__['MedicalImages'](root, task=source[0], class_index=class_index, split='uncertain',
                                                                                          download=True, transform=val_transform)
            without_uncertain_test_dataset = datasets.__dict__['MedicalImages'](root, task=source[0], class_index=class_index, split='test_without_uncertain',
                                                                                          download=True, transform=val_transform)
    return uncertain_dataset, without_uncertain_test_dataset

def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg

def multi_label_validate(val_loader, model, args, device, type="Target", mode='RGB', num_classes=6) -> float:
    batch_num = len(val_loader)
    print_freq = math.ceil(batch_num / 5)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    auc_meter_list = []
    class_map = {0:"ATL", 1:"CM",2:"PE",3:"CS", 4:"EDMA", 5:"PM",6:"Normal",7:"Average"}
    for i in range(num_classes+1):
        if i == num_classes:
            auc_meter_list.append(AverageMeter(class_map[7], ':3.1f'))
        else:
            auc_meter_list.append(AverageMeter(class_map[i], ':3.1f'))

    cls_accs = []
    cls_n_accs = []
    cls_p_accs = []
    class_map = {0:"ATL", 1:"CM",2:"PE",3:"CS", 4:"EDMA", 5:"PM",6:"Normal",7:"Average"}
    for i in range(num_classes+1):
        if i == num_classes:
            cls_accs.append(AverageMeter(class_map[7], ':3.1f'))
        else:
            cls_accs.append(AverageMeter(class_map[i], ':3.1f'))
    for i in range(num_classes+1):
        if i == num_classes:
            cls_n_accs.append(AverageMeter(class_map[7], ':3.1f'))
        else:
            cls_n_accs.append(AverageMeter(class_map[i], ':3.1f'))
    for i in range(num_classes+1):
        if i == num_classes:
            cls_p_accs.append(AverageMeter(class_map[7], ':3.1f'))
        else:
            cls_p_accs.append(AverageMeter(class_map[i], ':3.1f'))
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses] + [auc_meter_list[-1]]+auc_meter_list[0:-1],
        prefix=type+'Test: ')
    
    p_score_meter_list = []
    n_score_meter_list = []
    for i in range(num_classes):
        p_score_meter_list.append(AverageMeter(class_map[i]+' '+'p_score', ':3.1f'))
        n_score_meter_list.append(AverageMeter(class_map[i]+' '+'n_score', ':3.1f'))

    # switch to evaluate mode
    model.eval()
    cls_loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)
            if mode =='L':
                images = images.repeat(1, 3, 1, 1)

            # compute output
            output, f = model(images)
            target = torch.squeeze(target, 1).float()
            loss = cls_loss_function(output, target)

            # measure accuracy and record loss
            y_sigmoid = torch.sigmoid(output)
            for j in range(num_classes):
                y_tmp = y_sigmoid[:, j]
                p_score_avg = torch.mean(y_tmp[target[:,j] == 1])
                p_score_meter_list[j].update(p_score_avg.item(), y_tmp[target[:,j] == 1].shape[0])
                n_score_avg = torch.mean(y_tmp[target[:,j] == 0])
                n_score_meter_list[j].update(n_score_avg.item(), y_tmp[target[:,j] == 0].shape[0])

            cls_acc = multi_label_accuracy(output, target)
            for j in range(num_classes+1):
                cls_accs[j].update(cls_acc[j],images.size(0))
            # 正负例准确率
            for j in range(num_classes):
                p_acc,n_acc = binary_accuracy(output[:,j], target[:,j])
                if torch.isnan(p_acc) == False:
                    cls_p_accs[j].update(p_acc,images.size(0))
                if torch.isnan(n_acc) == False:
                    cls_n_accs[j].update(n_acc,images.size(0))


            cls_auc = multi_label_auc(output, target)
            if len(cls_auc)>0:
                for j in range(num_classes+1):
                    auc_meter_list[j].update(cls_auc[j], images.size(0))
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            
            if i % print_freq == 0:
                progress.display(i)

        print(' * Average AUC {top1.avg:.3f}'.format(top1=auc_meter_list[-1]))
        if confmat:
            print(confmat.format(args.class_names))

    res = []
    print('AUC')
    for i in range(num_classes+1):
        res.append(auc_meter_list[i].avg)
        print("{}".format(auc_meter_list[i].avg),end=" ")
    print("\n正例准确率")
    for i in range(num_classes):
        print("{}".format(cls_p_accs[i].avg),end=" ")
    print("\n负例准确率")
    for i in range(num_classes):
        print("{}".format(cls_n_accs[i].avg),end=" ")
    print('\n正例平均得分')
    for i in range(num_classes):
        print("{}".format(p_score_meter_list[i].avg),end=" ")
    print('\n负例平均得分')
    for i in range(num_classes):
        print("{}".format(n_score_meter_list[i].avg),end=" ")
    print("\n")
    return res, losses.avg



def data_explore(val_loader, model, args, device, type="Target", mode='RGB', num_classes=8):
    true_normal = 0
    bad_normal = 0
    pred_normal = 0
    pred_bad_normal = 0
    true_normal_pred_list = [0 for i in range(7)]
    true_multilabel = 0
    pred_true_multilabel = 0
    pred_multilabel = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)
            target = torch.squeeze(target, 1).float()
            output, f = model(images)
            output = torch.sigmoid(output)
            for j in range(target.size(0)):
                if target[j][-1] == 1:
                    if len(torch.nonzero(target[j]==1))==1:
                        true_normal += 1
                        for idex in torch.nonzero(output[j]>0.5):
                            true_normal_pred_list[idex.item()] += 1
                    else:
                        bad_normal += 1
            for j in range(output.size(0)):
                if output[j][-1] > 0.5:
                    if len(torch.nonzero(output[j]>0.5))==1:
                        pred_normal += 1
                    else:
                        pred_bad_normal += 1
            
            for j in range(target.size(0)):
                if len(torch.nonzero(target[j]==1)) > 1:
                    true_multilabel += 1
                    if len(torch.nonzero(output[j] > 0.5)) > 1:
                        pred_true_multilabel += 1
      
            for j in range(output.size(0)):
                if len(torch.nonzero(output[j]>0.5))==1:
                    pred_multilabel += 1
                   
    print("true normal:{}".format(true_normal))
    print("bad_normal:{}".format(bad_normal))
    print("pred true normal:{}".format(pred_normal))
    print("pred bad_normal:{}".format(pred_bad_normal))
    print(true_multilabel)
    print(pred_multilabel)
    print(pred_true_multilabel)
    print(true_normal_pred_list)

    



def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
