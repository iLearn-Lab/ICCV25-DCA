# """
# @author: Mingyang Liu
# @contact: mingyangliu1024@gmail.com
# """
# import argparse
# import builtins
# import math
# import os
# import random
# import shutil
# import time
# import warnings
# from tqdm import tqdm
# import numpy as np
# import faiss
# import torch.distributed as dist

# from typing import Optional
# import torch.nn as nn
# import torch
# import torch.nn.functional as F

# from tllib.alignment.my_supcon import SupConResNetMultiBottlenecks


# import random
# import time
# import math
# import warnings
# import argparse
# import shutil
# import os.path as osp
# from typing import Tuple

# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from matplotlib import pyplot as plt
# from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
# from torch.optim import SGD
# import torch.utils.data
# from torch.utils.data import DataLoader
# import torch.nn.functional as F

# import utils
# from tllib.alignment.my_supcon import SupConResNetMultiBottlenecks
# from tllib.modules.my_loss import SupConLoss
# from tllib.utils.data import ForeverDataIterator
# from tllib.utils.metric import accuracy, ConfusionMatrix, multi_label_auc, binary_accuracy, multi_label_accuracy
# from tllib.utils.meter import AverageMeter, ProgressMeter




# def compute_features_parallel(train_source_loader, G: nn.Module, bottlenecks: SupConResNetMultiBottlenecks, args, device,num_classes=6):
#     print('Computing features...')
#     G_parallel = torch.nn.DataParallel(G, device_ids=[0,1],dim=0)
#     G_parallel.eval()
#     bottlenecks_parallel = torch.nn.DataParallel(bottlenecks, device_ids=[0,1],dim=0)
#     bottlenecks_parallel.eval()
#     f_list = []
#     label_list = []
#     start = time.time()
#     with torch.no_grad():
#         for i, data in enumerate(train_source_loader):
#             images, target = data[:2]
#             images = images.to(device)
#             target = target.to(device)
#             target = torch.squeeze(target, 1).float() # b * 6
#             feat = bottlenecks_parallel(G_parallel(images)) # num_classes * b * 128
#             print(feat.shape)
#             f_list.append(feat)
#             label_list.append(target)
#     features = torch.cat(f_list, dim=1) # num_classes * num_images * 128
#     print(features.shape)
#     labels = torch.cat(label_list, dim=0)
#     print(labels.shape)
#     # dist.barrier()        
#     # dist.all_reduce(features, op=dist.ReduceOp.SUM)
#     print('Success')     
#     print('Compute features time: {}'.format(time.time()-start))
#     # return features.cpu(), labels.cpu()     
#     return features, labels



# def compute_features(train_source_loader, G: nn.Module, bottlenecks: SupConResNetMultiBottlenecks, args, device,num_classes=6):
#     print('Computing features...')
#     G.eval()
#     bottlenecks.eval()
#     f_list = []
#     label_list = []
#     start = time.time()
#     with torch.no_grad():
#         for i, data in enumerate(train_source_loader):
#             images, target = data[:2]
#             images = images.to(device)
#             target = target.to(device)
#             target = torch.squeeze(target, 1).float() # b * 6
#             feat = bottlenecks(G(images)) # num_classes * b * 128
#             f_list.append(feat)
#             label_list.append(target)
#     features = torch.cat(f_list, dim=1) # num_classes * num_images * 128
#     labels = torch.cat(label_list, dim=0)
#     print('Success')     
#     print('Compute features time: {}'.format(time.time()-start))
#     # return features.cpu(), labels.cpu()     
#     return features, labels



# def run_kmeans(f, num_classes, args, device):
#     """
#     Args:
#         x: data to be clustered  (b, f_dim)
#     """
    
#     print('performing kmeans clustering')
#     results = {'im2cluster':[],'centroids':[],'density':[]}
    
#     for seed, i in enumerate(range(num_classes)):
#         # intialize faiss clustering parameters
#         x = f[i]
#         d = x.shape[1]
#         k = 2
#         clus = faiss.Clustering(d, k)
#         clus.verbose = True
#         clus.niter = 20
#         clus.nredo = 5
#         clus.seed = seed
#         clus.max_points_per_centroid = 1000
#         clus.min_points_per_centroid = 10

#         res = faiss.StandardGpuResources()
#         cfg = faiss.GpuIndexFlatConfig()
#         cfg.useFloat16 = False
#         cfg.device = args.gpu   
#         index = faiss.GpuIndexFlatL2(res, d, cfg)  

#         clus.train(x, index)   

#         D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
#         im2cluster = [int(n[0]) for n in I]
        
#         # get cluster centroids
#         centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
#         # sample-to-centroid distances for each cluster 
#         Dcluster = [[] for c in range(k)]          
#         for im,i in enumerate(im2cluster):
#             Dcluster[i].append(D[im][0])
        
#         # concentration estimation (phi)        
#         density = np.zeros(k)
#         for i,dist in enumerate(Dcluster):
#             if len(dist)>1:
#                 d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
#                 density[i] = d     
                
#         #if cluster only has one point, use the max to estimate its concentration        
#         dmax = density.max()
#         for i,dist in enumerate(Dcluster):
#             if len(dist)<=1:
#                 density[i] = dmax 

#         density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
#         density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
#         # convert to cuda Tensors for broadcast
#         centroids = torch.Tensor(centroids).cuda()
#         centroids = nn.functional.normalize(centroids, p=2, dim=1)    

#         im2cluster = torch.LongTensor(im2cluster).cuda()               
#         density = torch.Tensor(density).cuda()
        
#         results['centroids'].append(centroids)
#         results['density'].append(density)
#         results['im2cluster'].append(im2cluster)    
        
#     return results

# def sup_kmeans(f,labels):
#     # 每一类
#     # print("Computing Centroids...")
#     with torch.no_grad():
#         centroids = []
#         for i in range(f.shape[0]):
#             x = f[i] # images_sum * 128
#             # print('%%%%%%%%%%%%%%%%%%%%%%%%')
#             # print(x.shape)
#             label = labels[:,i]
#             # print(label.shape)
#             p_sum = label.sum()
#             n_sum = x.shape[0] - p_sum
#             # print(p_sum)
#             # print(n_sum)
#             label = label.view(-1,1).repeat(1, x.shape[1])
#             # print(label.shape) # images_sum * 128
#             center_p = torch.sum(x*label,dim=0) / p_sum # 正例中心
#             center_n = torch.sum(x*(1.0-label),dim=0) / n_sum # 负例中心
#             centroids.append((center_p,center_n))
#     # print("Success")
#     return centroids


    
# def binary_accuracy_hardlabel(output: torch.Tensor, target: torch.Tensor, threshold=0.5):
#     with torch.no_grad():
#         positive_accuracy = torch.mean((output[target == 1] == 1).float())
#         negative_accuracy  = torch.mean((output[target == 0] == 0).float())
    
#         return positive_accuracy, negative_accuracy

# # def predict(val_loader: DataLoader, G: nn.Module, bottlenecks: SupConResNet,centroids,args,device,num_classes):
#     batch_num = len(val_loader)
#     print_freq = math.ceil(batch_num / 5)
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':6.2f')
#     auc_meter_list = []
#     class_map = {0:"ATL", 1:"CM",2:"PE",3:"CS", 4:"EDMA", 5:"PM",6:"Normal",7:"Average"}
#     for i in range(num_classes+1):
#         if i==num_classes:
#             auc_meter_list.append(AverageMeter(class_map[7], ':3.1f'))
#         else:
#             auc_meter_list.append(AverageMeter(class_map[i], ':3.1f'))
#     cls_accs = []
#     cls_n_accs = []
#     cls_p_accs = []
#     class_map = {0:"ATL", 1:"CM",2:"PE",3:"CS", 4:"EDMA", 5:"PM",6:"Normal",7:"Average"}
#     for i in range(num_classes+1):
#         if i==num_classes:
#             cls_accs.append(AverageMeter(class_map[7], ':3.1f'))
#         else:
#             cls_accs.append(AverageMeter(class_map[i], ':3.1f'))
#     for i in range(num_classes):
#         cls_n_accs.append(AverageMeter(class_map[i], ':3.1f'))
#         cls_p_accs.append(AverageMeter(class_map[i], ':3.1f'))
#     progress = ProgressMeter(
#         batch_num,
#         [batch_time, losses]+[auc_meter_list[-1]]+auc_meter_list[0:-1],
#         prefix="Test: ")
    
#     cls_loss_function = nn.BCEWithLogitsLoss(reduction="mean")
#     # switch to evaluate mode
#     G.eval()
#     F.eval()

#     with torch.no_grad():
#         end = time.time()
#         for i, data in enumerate(val_loader):
#             images, target = data[:2]
#             images = images.to(device)
#             target = target.to(device)
#             target = torch.squeeze(target, 1).float()
#             f = bottlenecks(G(images))
    
#             cluster_label_list = []
#             for j in num_classes:
#                 x = f[j] # b * 128
#                 center_p, center_n = centroids[j] # 128
#                 center  = torch.stack([center_n,center_p],dim=0)  # 2*128 [-,+]
#                 distances = torch.div(
#                     torch.matmul(x, center.T),
#                     args.temperature)
#                 cluster_label = torch.argmin(torch.tensor(distances), dim=1) # b
#                 cluster_label_list.append(cluster_label)
#             cluster_labels = torch.stack(cluster_label_list,dim=1) # b * num_classes  hard label

#             for j in range(num_classes):
#                 p_acc,n_acc = binary_accuracy(cluster_labels[:,j], target[:,j])
#                 if torch.isnan(p_acc) == False:
#                     cls_p_accs[j].update(p_acc,images.size(0))
#                 if torch.isnan(n_acc) == False:
#                     cls_n_accs[j].update(n_acc,images.size(0))

# def predict_hard_labels(f,centroids, args, device):
#     # 返回的是硬标签
#     f = f.detach()
#     num_classes = f.shape[0]
#     cluster_label_list = []
#     for j in range(num_classes):
#         x = f[j].to(device) # b * 128
#         center_p, center_n = centroids[j] # 128
#         center  = torch.stack([center_n,center_p],dim=0).to(device)  # 2*128 [-,+]
#         product = torch.div(torch.matmul(x, center.T), args.temperature)  # b * 2 [-,+]
#         cluster_label = torch.argmax(product, dim=1) # b
#         cluster_label_list.append(cluster_label)
#     cluster_labels = torch.stack(cluster_label_list,dim=1) # b * num_classes  hard label
#     return cluster_labels


# def predict_soft_labels(f,centroids, args, device):
#     # 返回的是软标签
#     # python函数传参是一个局部变量还是地址呀
#     f = f.detach()
#     num_classes = f.shape[0]
#     soft_label_list = []
#     for j in range(num_classes):
#         x = f[j].to(device) # b * 128
#         center_p, center_n = centroids[j] # 128
#         center  = torch.stack([center_n,center_p],dim=0).to(device)  # 2*128 [-,+]
#         product = torch.div(torch.matmul(x, center.T), 1)  # b * 2 [-,+]
#         theta = torch.acos(product)
#         # print(theta)
#         p_score = theta[:,0] / theta.sum(1)
#         n_score = torch.tensor(1.) - p_score
#         score = torch.cat([torch.div(p_score, 0.07).view(-1,1), torch.div(n_score,0.07).view(-1,1)],dim=1)
#         # print(p_score)
#         score = torch.softmax(score,dim=1)
#         # print(score)
#         soft_label_list.append(score[:,0]) # 检查是否是p_score (b)
#     soft_label = torch.stack(soft_label_list,dim=1)   # (b, num_classes)
#     return soft_label


