import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('/home/musa/home/musa/models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from models.BBSNet_model import BBSNet
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from scipy import ndimage


# set the device for training
# set loss function
# CE = torch.nn.BCEWithLogitsLoss()
def deal_batch(mask):
    gt = []
    batch = mask.size()[0]
    for i in range(1, batch + 1):
        GT = mask[i - 1:i]
        GT_edge_enhanced = ndimage.gaussian_laplace(np.squeeze(GT.cpu().detach().numpy()), sigma=5)
        GT_edge1 = torch.tensor(np.int64(GT_edge_enhanced < -0.001))
        GT_edge2 = torch.tensor(np.int64(GT_edge_enhanced > 0.001))
        GT_EE = GT_edge2 + GT_edge1
        GT_EE = torch.unsqueeze(GT_EE, dim=0)
        gt.append(torch.unsqueeze(GT_EE, dim=0))
    num = len(gt)
    if num == 8:
        gt_enhance = torch.cat([gt[0], gt[1], gt[2], gt[3], gt[4], gt[5], gt[6], gt[7]], 0)
    if num ==1:
        gt_enhance = gt[0]
    else: gt_enhance = gt[2]
    return gt_enhance



def CE(pred, mask):
    """
    Introduction:
        Hybrid Eloss
    Paramaters:
        :param pred:
        :param mask:
        :return:
    Usage:
        loss = hybrid_e_loss(prediction_map, gt_mask)
        loss.backward()
    """
    # adaptive weighting masks
    # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weit = 1 + deal_batch(mask).to(torch.float32).cuda()
    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)
    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred
    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask
    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))
    # weighted iou loss function
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)
    return (wbce  + wiou).mean()


def CE1(pred, mask):
    """
    final s
    Introduction:
        Hybrid Eloss
    Paramaters:
        :param pred:
        :param mask:
        :return:
    Usage:
        loss = hybrid_e_loss(prediction_map, gt_mask)
        loss.backward()
    """
    # adaptive weighting masks
    # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weit = 1 + deal_batch(mask).to(torch.float32).cuda()
    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    # wbce = ((wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)
    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred
    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask
    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))
    # weighted iou loss function
    inter = ((pred * mask) ).sum(dim=(2, 3))
    union = ((pred + mask) ).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)
    return (wbce).mean()
# import torch.nn as nn
# model = nn.DataParallel(model)
# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            
            depths = depths.cuda()
            s1, s2, s3, s4, s5, s6,s7 = model(images, depths)
            
            loss5 = CE(s5, gts)
            loss6 = CE(s6, gts)
            loss7 = CE(s7, gts)
            loss = loss5 + loss6 + loss7
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss6: {:.4f} Loss7: {:0.4f} Loss5: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss6.data, loss7.data, loss5.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                             format(epoch, opt.epoch, i, total_step, loss6.data, loss7.data))
                # writer.add_scalar('Loss', loss.data, global_step=step)
                # grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('RGB', grid_image, step)
                # grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('Ground_truth', grid_image, step)
                # res=s1[0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('s1', torch.tensor(res), step,dataformats='HW')
                # res=s2[0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('s2', torch.tensor(res), step,dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'BBSNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'BBSNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            _, _, _, _, _, _, res = model(image, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'BBSNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 1')
    cudnn.benchmark = True
    # build the model
    model = BBSNet()
    if (opt.load is not None):
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)
    model = torch.nn.DataParallel(model.cuda())
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    # set the path
    image_root = opt.rgb_root
    gt_root = opt.gt_root
    depth_root = opt.depth_root
    test_image_root = opt.test_rgb_root
    test_gt_root = opt.test_gt_root
    test_depth_root = opt.test_depth_root
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # load data
    print('load data...')
    train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize,
                              trainsize=opt.trainsize)
    test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
    total_step = len(train_loader)
    logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("BBSNet-Train")
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
            opt.decay_epoch))
    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)