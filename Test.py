import torch
import torch.nn.functional as F
import sys
sys.path.append('/home/musa/home/musa/models')
import numpy as np
import os, argparse
import cv2
from models.BBSNet_model import BBSNet
from data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/musa/home/musa/dataset/RGBT_for_test/',help='test dataset path')
parser.add_argument('--model_id',type=str,default='50',help='model id')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 1')
# pred = torch.load('./BBSNet_cpts1/BBSNet_epoch_best.pth')
#load the model
model = BBSNet()
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
# model.load_state_dict(torch.load('./BBSNet_cpts1/BBSNet_epoch_2.pth'))
pred = torch.load('./BBSNet_cpts_new/BBSNet_epoch_best.pth')

temp = model.state_dict()
for k ,v in pred.items():
    name = k.replace('module.', '', 1);
    temp[name] = v;
import torch.nn as nn
model.load_state_dict(temp)
model = nn.DataParallel(model)


model.cuda()
model.eval()

#test
test_datasets = ['VT5000','VT1000','VT821',]
for dataset in test_datasets:
    save_path = './test_maps/EANet_epoch_'+opt.model_id+'/'+dataset+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    t_root=dataset_path +dataset +'/T/'
    test_loader = test_dataset(image_root, gt_root,t_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,t, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        t = t.cuda()
        _, _,_, _,res1,res2, res = model(image,t)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
