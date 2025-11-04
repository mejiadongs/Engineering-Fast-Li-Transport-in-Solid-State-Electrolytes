import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet
from sklearn.model_selection import KFold
import pandas as pd

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--radius', default=5.0, type=float,
                    help='neighbor search radius (default: 5.0 Å)')

train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def save_cv_results(fold_mae_errors, fold_r2_scores):
    # 保存交叉验证结果到CSV文件
    cv_results = pd.DataFrame({
        'Fold': range(1, len(fold_mae_errors) + 1),
        'MAE': fold_mae_errors,
        'R2': fold_r2_scores
    })
    cv_results.to_csv('cross_validation_results.csv', index=False)
    print("Cross-validation results saved to 'cross_validation_results.csv'")

def plot_cv_results(fold_mae_errors, fold_r2_scores):
    # 创建折数列表
    folds = range(1, len(fold_mae_errors) + 1)
    
    # 创建条形图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.bar(folds, fold_mae_errors)
    ax1.set_title('MAE across folds')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('MAE')
    
    ax2.bar(folds, fold_r2_scores)
    ax2.set_title('R2 across folds')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('R2')
    
    plt.tight_layout()
    plt.savefig('cross_validation_barplots.png')
    plt.close()
    print("Cross-validation bar plots saved to 'cross_validation_barplots.png'")

def plot_cv_trend(fold_mae_errors, fold_r2_scores):
    # 绘制MAE和R2的趋势图
    folds = range(1, len(fold_mae_errors) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('MAE', color=color)
    ax1.plot(folds, fold_mae_errors, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('R2', color=color)
    ax2.plot(folds, fold_r2_scores, color=color, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('MAE and R2 across folds')
    plt.tight_layout()
    plt.savefig('cross_validation_trend.png')
    plt.close()
    print("Cross-validation trend plot saved to 'cross_validation_trend.png'")

def main():
    global args, best_mae_error

    # Load dataset
    dataset = CIFData(*args.data_options, radius=args.radius)
    collate_fn = collate_pool
    
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=50, shuffle=True, random_state=42)
    
    # Lists to store performance metrics for each fold
    fold_mae_errors = []
    fold_r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        print(f"Fold {fold}")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(dataset, train_idx),
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.workers,
            pin_memory=args.cuda
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(dataset, val_idx),
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.workers,
            pin_memory=args.cuda
        )
        
        # Initialize model, loss function, optimizer, and scheduler
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    classification=args.task == 'classification')
        
        if args.cuda:
            model.cuda()
        
        criterion = nn.MSELoss() if args.task == 'regression' else nn.NLLLoss()
        
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
        
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                gamma=0.1)

        # Initialize normalizer
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

        # Train the model
        best_mae_error = float('inf')
        for epoch in range(args.epochs):
            train(train_loader, model, criterion, optimizer, epoch, normalizer)
            mae_error = validate(val_loader, model, criterion, normalizer)
            
            # Save the latest checkpoint (overwriting previous epoch checkpoint)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=f'checkpoint_fold_{fold}.pth.tar')
            
            # Save the best model if it achieves a lower MAE than previous best
            if mae_error < best_mae_error:
                best_mae_error = mae_error
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mae_error': best_mae_error,
                    'optimizer': optimizer.state_dict(),
                }, is_best=True, filename=f'model_best_fold_{fold}.pth.tar')
            
            scheduler.step()

        # Evaluate the best model on the validation set
        mae_error = validate(val_loader, model, criterion, normalizer, test=True)
        r2 = calculate_r2(val_loader, model, normalizer)
        
        fold_mae_errors.append(mae_error)
        fold_r2_scores.append(r2)
        
        print(f"Fold {fold} - MAE: {mae_error:.4f}, R2: {r2:.4f}")
    
    # Calculate average performance
    avg_mae = np.mean(fold_mae_errors)
    avg_r2 = np.mean(fold_r2_scores)
    
    print(f"Average MAE across all folds: {avg_mae:.4f}")
    print(f"Average R2 across all folds: {avg_r2:.4f}")
    
    # Save and visualize cross-validation results
    save_cv_results(fold_mae_errors, fold_r2_scores)
    plot_cv_results(fold_mae_errors, fold_r2_scores)
    plot_cv_trend(fold_mae_errors, fold_r2_scores)
    
    # Final evaluation and plotting
    all_targets = []
    all_predictions = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        val_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(dataset, val_idx),
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.workers,
            pin_memory=args.cuda
        )
        
        # Use the last trained model for predictions
        targets, predictions = get_predictions(val_loader, model, normalizer)
        all_targets.extend(targets)
        all_predictions.extend(predictions)

    overall_r2 = r2_score(all_targets, all_predictions)
    plot_predictions_vs_true_with_r2(all_targets, all_predictions, overall_r2)
    residuals = np.array(all_targets) - np.array(all_predictions)
    plot_residuals(all_targets, residuals)
    plot_error_distribution(residuals)
    
    
def get_predictions(data_loader, model, normalizer):
    model.eval()
    targets = []
    predictions = []
    
    for input, target, _ in data_loader:
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        
        output = model(*input_var)
        
        targets.extend(target.numpy())
        predictions.extend(normalizer.denorm(output.data.cpu()).numpy())
    
    return targets, predictions
    
def calculate_r2(data_loader, model, normalizer):
    model.eval()
    true_values = []
    pred_values = []
    
    for input, target, _ in data_loader:
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        
        output = model(*input_var)
        
        true_values.extend(target.numpy())
        pred_values.extend(normalizer.denorm(output.data.cpu()).numpy())
    
    return r2_score(true_values, pred_values)

# 修改 train 和 validate 函数以接受 normalizer 参数
def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )

def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])

            if args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            if args.task == 'regression':
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                if test:
                    test_pred = normalizer.denorm(output.data.cpu())
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))
                if test:
                    test_pred = torch.exp(output.data.cpu())
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label, auc=auc_scores))
        return auc_scores.avg
            
class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def plot_loss_curve(train_losses, val_losses):
    # 保存训练和验证损失到 CSV 文件
    with open('loss_curve_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([epoch, train_loss, val_loss])
    
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

def plot_predictions_vs_true_with_r2(targets, predictions, r2):
    # 保存预测值与真实值的比较数据到 CSV 文件
    with open('predictions_vs_true_with_r2_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True Values', 'Predicted Values'])
        for target, prediction in zip(targets, predictions):
            writer.writerow([target, prediction])
    
    plt.figure()
    plt.scatter(targets, predictions, label='Predictions vs True Values')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values (R² = {r2:.2f})')
    plt.legend()
    plt.savefig('predictions_vs_true_with_r2.png')
    plt.show()

def plot_residuals(targets, residuals):
    # 保存残差数据到 CSV 文件
    with open('residuals_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True Values', 'Residuals (True - Predicted)'])
        for target, residual in zip(targets, residuals):
            writer.writerow([target, residual])
    
    plt.figure()
    plt.scatter(targets, residuals, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('True Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residuals vs True Values')
    plt.legend()
    plt.savefig('residuals_vs_true.png')
    plt.show()
    
def plot_error_distribution(residuals):
    # 保存误差分布数据到 CSV 文件
    with open('error_distribution_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Residuals'])
        for residual in residuals:
            writer.writerow([residual])
    
    plt.figure()
    sns.histplot(residuals, bins=30, kde=True, edgecolor='k', color='blue', alpha=0.6)
    plt.xlabel('Prediction Error (True - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution with KDE')
    plt.savefig('error_distribution_with_kde.png')
    plt.show()
    
def save_error_distribution_data(residuals, filename='error_distribution_data.csv'):
    try:
        # 确保 residuals 是一维数组
        residuals = np.array(residuals).flatten()
        
        df = pd.DataFrame({'Residuals': residuals})
        df.to_csv(filename, index=False)
        print(f"Error distribution data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")
        print(f"Shape of residuals: {np.array(residuals).shape}")
        print(f"Type of residuals: {type(residuals)}")

def plot_error_distribution(residuals):
    try:
        # 确保 residuals 是一维数组
        residuals = np.array(residuals).flatten()
        
        # 保存误差分布数据到 CSV 文件
        save_error_distribution_data(residuals)
        
        # 创建一个 DataFrame
        df = pd.DataFrame({'Residuals': residuals})
        
        # 创建一个新的图形，包含两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制箱线图
        sns.boxplot(y='Residuals', data=df, ax=ax1)
        ax1.set_title('Error Distribution (Box Plot)')
        ax1.set_ylabel('Prediction Error (True - Predicted)')
        
        # 绘制小提琴图
        sns.violinplot(y='Residuals', data=df, ax=ax2)
        ax2.set_title('Error Distribution (Violin Plot)')
        ax2.set_ylabel('Prediction Error (True - Predicted)')
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('error_distribution_box_violin.png')
        print("Error distribution plot saved as 'error_distribution_box_violin.png'")
        plt.close()  # 关闭图形，避免在非交互环境中显示
    except Exception as e:
        print(f"Error plotting distribution: {e}")
        print(f"Shape of residuals: {np.array(residuals).shape}")
        print(f"Type of residuals: {type(residuals)}")
    
if __name__ == '__main__':
    main()
