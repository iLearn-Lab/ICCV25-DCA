import torch
import prettytable
from sklearn.metrics import roc_auc_score
import numpy as np

__all__ = ['keypoint_detection']

def binary_accuracy_original(output: torch.Tensor, target: torch.Tensor, threshold=0.5) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= threshold).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

# 计算二分类正例和负例的准确率
# 如果没有正样本，返回值为nan
def binary_accuracy(output: torch.Tensor, target: torch.Tensor, threshold=0.5):
    with torch.no_grad():
        prediction = torch.sigmoid(output)
        prediction = torch.where(prediction>threshold, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
      
        positive_accuracy = torch.mean((prediction[target == 1] == 1).float())
        negative_accuracy  = torch.mean((prediction[target == 0] == 0).float())
    
        return positive_accuracy, negative_accuracy



def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def multi_label_accuracy(output, target, threshold=0.5):
    """
    计算每类的二分类acc和平均acc
    output b*c
    target b*c
    Returns:
    acc_list=[acc_1, acc_2,……,acc_average]
    """
    with torch.no_grad():
        batch_size, c = output.shape
        prediction = torch.sigmoid(output)
        zero = torch.zeros_like(prediction)
        one = torch.ones_like(prediction)
        prediction = torch.where(prediction>threshold, one,zero)
        acc_matrix = torch.eq(prediction, target)
        acc_list = torch.sum(acc_matrix, dim=0) * (100.0 / batch_size)
        acc_average = torch.sum(acc_list) / c
        res = acc_list.cpu().numpy().tolist()
    res.append(acc_average.item())
    return res


def multi_label_auc(output, target):
    with torch.no_grad():
        output_array = output.cpu().numpy()
        target_array = target.cpu().numpy()
        try:
            res  = (roc_auc_score(target_array,output_array,average=None)*100.0).tolist() ## y_true=ground_truth
            average = sum(res)/len(res)
            res.append(average)
        except ValueError:
            res = []
        # res  = (roc_auc_score(target_array,output_array,average=None)*100.0).tolist() ## y_true=ground_truth
        # average = sum(res)/len(res)
        # res.append(average)
    
    return res
    
def auc(output, target):
    with torch.no_grad():
        output_array = output.cpu().numpy()
        target_array = target.cpu().numpy()
        try:
            res  = [roc_auc_score(target_array,output_array,average=None)*100.0]## y_true=ground_truth
        except ValueError:
            res = []
    return res



class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        """
        Update confusion matrix.

        Args:
            target: ground truth
            output: predictions of models

        Shape:
            - target: :math:`(minibatch, C)` where C means the number of classes.
            - output: :math:`(minibatch, C)` where C means the number of classes.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """compute global accuracy, per-class accuracy and per-class IoU"""
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

    def format(self, classes: list):
        """Get the accuracy and IoU for each class in the table format"""
        acc_global, acc, iu = self.compute()

        table = prettytable.PrettyTable(["class", "acc", "iou"])
        for i, class_name, per_acc, per_iu in zip(range(len(classes)), classes, (acc * 100).tolist(), (iu * 100).tolist()):
            table.add_row([class_name, per_acc, per_iu])

        return 'global correct: {:.1f}\nmean correct:{:.1f}\nmean IoU: {:.1f}\n{}'.format(
            acc_global.item() * 100, acc.mean().item() * 100, iu.mean().item() * 100, table.get_string())

