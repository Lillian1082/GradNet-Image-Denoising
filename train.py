import os
import argparse
import torchvision

import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.models import *
from math import log10
from utilsData import *
from utils import *

os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="PDNet")
parser.add_argument('--archi', type=str, default="dncnn_sk_conGradient_before", help='use DnCNN as reference?')
parser.add_argument('--resume', type=int, default=0, help='train or finetune?')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--device_ids", type=int, default=1, help="move to GPU")
parser.add_argument("--numOfLayers", type=int, default=20, help="Number of total layers")

parser.add_argument("--pretrained_path", type=str, default="", help='path of pretrained checkpoints')
parser.add_argument("--pretrained_model", type=str, default="", help='path of pretrained checkpoints')
parser.add_argument("--pretrained_archi", type=str, default="", help='archi of pretrained model')

parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--train_dataPath", type=str, default='data/Synthetic_set/train/GT', help='path of training files to process') #data/BSDS500_new/train
parser.add_argument("--test_dataPath", type=str, default='data/Synthetic_set/val/CBSD68', help='path of validation files to process') #data/BSDS500_new/CBSD68
parser.add_argument("--noiseLevel", type=int, default=15, help="adjustable noise level")
parser.add_argument("--outf", type=str, default="logs/try", help='path of log files')
# parser.add_argument('--crop', default=False, type=bool, help='crop patches?')
parser.add_argument('--cropSize', default=64, type=int,  help='crop patches? training images crop size')
parser.add_argument('--real', default=0, type=bool, help='Real Dataset?')
parser.add_argument('--seed', default=0, type= int, help='seed of all random')
parser.add_argument('--randomCount', default=1, type= int, help='the number of patches got from each patch')
parser.add_argument('--augment', default=True, type= bool, help='whether to apply data augmentation to it')
parser.add_argument('--grad_weight', default=0.1, type= float, help='weight of gradient loss')
opt = parser.parse_args()

torch.backends.cudnn.enabled = False # will make the speed slow, cudnn could speedup the training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)

def main():
    # Load dataset
    print('Loading dataset ...\n')
    start = time.time()
    DDataset = Dataset_Grad(opt.train_dataPath, randomCount=opt.randomCount, augment=opt.augment, cropPatch=True,
                             cropSize=opt.cropSize, real=0, noiseLevel=opt.noiseLevel)
    loaderTr = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=opt.batchSize, shuffle=True)
    VDataset = Dataset_Grad(opt.test_dataPath, randomCount=1, augment=0, cropPatch=0,
                             cropSize=opt.cropSize, real=0, noiseLevel=opt.noiseLevel)
    loaderVal = DataLoader(dataset=VDataset, num_workers=4, batch_size=1, shuffle=False)


    end = time.time()
    print (round(end - start, 7))    
    print("# of training samples: %d\n\n" % int(len(loaderTr)))
    
    # Build model
    if opt.archi == 'dncnn':
        net = DnCNN(channels=3, num_of_layers=opt.numOfLayers)
        Loss_criterion = nn.L1Loss(reduction='sum')

    if opt.archi == 'dncnn_sk':
        net = DnCNN_sk(channels=3, num_of_layers=opt.numOfLayers)
        Loss_criterion = nn.L1Loss(reduction='sum')

    if opt.archi == 'dncnn_sk_conGradient':
        net = DnCNN_sk_conGradient(channels=3, num_of_layers=opt.numOfLayers)
        Loss_criterion = nn.L1Loss(reduction='sum')

    if opt.archi == 'dncnn_sk_conGradient_before':
        net = DnCNN_sk_conGradient_before(channels=3, num_of_layers=opt.numOfLayers)
        Loss_criterion = nn.L1Loss(reduction='sum')

    if opt.archi == 'dncnn_sk_conGradient_onimage':
        net = DnCNN_conGradient_image(channels=3, num_of_layers=opt.numOfLayers)
        Loss_criterion = nn.L1Loss(reduction='sum')

    if opt.archi == 'Resnet_sk':
        net = Resnet_sk(channels=3)
        Loss_criterion = nn.L1Loss(reduction='sum')

    if opt.archi == 'Resnet_sk_conGradient_before':
        net = Res_conGradient_before(channels=3)
        Loss_criterion = nn.L1Loss(reduction='sum')

    if opt.archi == 'Resnet_sk_conGradient_onimage':
        net = Res_conGradient_onimage(channels=3)
        Loss_criterion = nn.L1Loss(reduction='sum')

    if opt.archi == 'Resnet_sk_conGradient':
        net = Res_conGradient(channels=3)
        Loss_criterion = nn.L1Loss(reduction='sum')

    net.apply(weights_init_kaiming)

    # Move to GPU
    device_ids = range(opt.device_ids)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    Loss_criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.resume == 1 :
        model.load_state_dict(torch.load(os.path.join(opt.outf, 'checkpoint', 'net_4.pth')))

    # training
    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)
        os.mkdir(os.path.join(opt.outf, 'checkpoint'))
        os.mkdir(os.path.join(opt.outf, 'test_output'))

    writer = SummaryWriter(opt.outf)

    best_PSNR = 0
    best_ssim = 0
    best_epoch = 0
    last_epoch = 0
    for epoch in range(opt.epochs):
        if epoch == opt.milestone:
            last_epoch = epoch
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"]/5
        elif (epoch - last_epoch) == 20:
            last_epoch = epoch
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"]/5
        for param_group in optimizer.param_groups:
            print('lr', param_group["lr"], 'last_epoch', last_epoch, 'best_epoch', best_epoch)
            if (param_group["lr"]<1e-6):
                assert 0

        train_avg_loss = 0
        train_avg_psnr = 0
        train_avg_ssim = 0
        for i, data in enumerate(loaderTr, 1):
            # training step
            model.train()
            model.zero_grad()            
            optimizer.zero_grad()

            imgClear, imgNoisy = data[0], data[1] # Dataset_BSDS500_4
            imgClear, imgNoisy = imgClear.cuda(), imgNoisy.cuda()
            imgDiff = imgNoisy - imgClear

            if (opt.archi == 'dncnn_sk_conGradient') or (opt.archi == 'dncnn_sk_conGradient_onimage') or (opt.archi =='dncnn_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_onimage') or (opt.archi == 'Resnet_sk_conGradient'):
                # grad = img_gradient_total(imgClear)
                grad = img_gradient_total(imgNoisy)
                # imgDenoised = get_intermediate_result(imgNoisy, opt.device_ids, opt.pretrained_archi,
                #                                       opt.pretrained_path, opt.pretrained_model)
                # grad = img_gradient_total(imgDenoised)
                # print('grad', grad)
                # torchvision.utils.save_image(imgDenoised[0], 'imgDenoised.jpg')
                # torchvision.utils.save_image((grad[0]-grad[0].min())/(grad[0].max()-grad[0].min()), 'grad.jpg')
                outRes = model(imgNoisy, grad)
            else:
                outRes = model(imgNoisy)

            # print('Noise:', imgDiff.max(), imgDiff.min(), imgDiff.sum(), imgDiff)
            # psnrTr = batchPSNR(imgNoisy, imgClear, 1.)
            # print('InputPSNR:', psnrTr)
            # # torchvision.utils.save_image(imgClear[0], 'imgClear_%d.jpg'%i)
            # # torchvision.utils.save_image(imgNoisy[0], 'imgNoisy_%d.jpg'%i)
            # if i==100:
            #     assert 0

            # Gradient_Weight
            # outRes = outRes*grad
            # grad = img_gradient_total(imgClear)
            # denoise_loss = torch.mul(torch.abs(outRes - imgClear), torch.abs(grad)).sum()

            # # L1_loss
            denoise_loss = Loss_criterion(outRes, imgClear) #output clean image
            # denoise_loss = Loss_criterion(outRes, imgDiff)

            # Gradient_loss
            gradient_loss = torch.tensor(0)
            # clear_x, clear_y = img_gradient(imgClear)
            # x_grad, y_grad = img_gradient(outRes)
            # gradient_loss = Loss_criterion(x_grad, clear_x) + Loss_criterion(y_grad, clear_y)

            # Total loss
            loss = denoise_loss + opt.grad_weight*gradient_loss

            loss.backward()
            optimizer.step()

            model.eval()
            # results
            imgResult = torch.clamp(outRes, 0., 1.)
            # imgResult = torch.clamp((imgNoisy-outRes), 0., 1.) #outRes
            psnrTr = batchPSNR(imgResult, imgClear, 1.)
            score_ssim = batchSSIM(imgClear, imgResult, win_size=3, multichannel=True)
            train_avg_loss += loss.item()
            train_avg_psnr += psnrTr
            train_avg_ssim += score_ssim
            print("[epoch %d][%d/%d] denoise_loss: %.4f gradient_loss: %.4f loss: %.4f PSNRTr: %.4f SSIMTr: %.4f" %
                (epoch+1, i, len(loaderTr), denoise_loss.item(), gradient_loss.item(), loss.item(), psnrTr, score_ssim)) #semantic_loss: %.4f semantic_loss.item()

            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
        train_avg_loss /= len(loaderTr)
        train_avg_psnr /= len(loaderTr)
        train_avg_ssim /= len(loaderTr)
        print("[epoch %d] Avg_denoise_loss: %.4f Avg_PSNRTr: %.4f SSIMTr: %.4f" %
              (epoch + 1, train_avg_loss, train_avg_psnr, train_avg_ssim))
        writer.add_scalar('loss', train_avg_loss, epoch)
        writer.add_scalar('PSNR on training data', train_avg_psnr, epoch)
        writer.add_scalar('SSIM on training data', train_avg_ssim, epoch)

        torch.cuda.empty_cache()
        model.eval()
        # validation
        val_avg_loss = 0
        val_input_psnr = 0
        val_avg_psnr = 0
        val_avg_ssim = 0
        with torch.no_grad():
            for i, data in enumerate(loaderVal, 0):
                imgClear, imgNoisy = data[0], data[1]  # Dataset_BSDS500_4
                imgClear, imgNoisy = imgClear.cuda(), imgNoisy.cuda()
                imgDiff = imgNoisy - imgClear

                if (opt.archi == 'dncnn_sk_conGradient') or (opt.archi == 'dncnn_sk_conGradient_onimage') or (opt.archi == 'dncnn_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_onimage') or (opt.archi == 'Resnet_sk_conGradient'):
                    # grad = img_gradient_total(imgClear)
                    grad = img_gradient_total(imgNoisy)
                    # imgDenoised = get_intermediate_result(imgNoisy, opt.device_ids, opt.pretrained_archi,
                    #                                       opt.pretrained_path, opt.pretrained_model) # Dataset_BSDS500_4
                    # grad = img_gradient_total(imgDenoised)
                    outRes = model(imgNoisy, grad)
                else:
                    outRes = model(imgNoisy)
                denoise_loss = Loss_criterion(outRes, imgClear) # Clean image
                # denoise_loss = Loss_criterion(outRes, imgDiff) # Noise
                imgResult = torch.clamp(outRes, 0., 1.) # Clean image
                # imgResult = torch.clamp((imgNoisy-outRes), 0., 1.) # Noise
                psnrInput = batchPSNR(imgNoisy, imgClear, 1.)
                psnrVal = batchPSNR(imgResult, imgClear, 1.)
                score_ssim_val = batchSSIM(imgClear, imgResult, win_size=3, multichannel=True)
                val_avg_loss += denoise_loss.item()
                val_input_psnr += psnrInput
                val_avg_psnr += psnrVal
                val_avg_ssim += score_ssim_val

                print("[epoch %d][%d/%d] val_denoise_loss: %.4f psnrInput: %.4f psnrVal: %.4f SSIM: %.4f" %
                      (epoch + 1, (i+1), len(loaderVal), denoise_loss.item(), psnrInput, psnrVal, score_ssim_val)) # val_semantic_loss: %.4f  semantic_loss.item()

            val_avg_loss /= len(loaderVal)
            val_avg_psnr /= len(loaderVal)
            val_avg_ssim /= len(loaderVal)
            print("[epoch %d] Avg_denoise_loss: %.4f Avg_PSNRTr: %.4f SSIM: %.4f" %
                  (epoch + 1, val_avg_loss, val_avg_psnr, val_avg_ssim))
            writer.add_scalar('loss', val_avg_loss, epoch)
            writer.add_scalar('PSNR on testing data', val_avg_psnr, epoch)
            writer.add_scalar('SSIM on testing data', val_avg_ssim, epoch)

        # save model
        if val_avg_psnr > best_PSNR:
            best_epoch = epoch+1
            last_epoch = best_epoch
            best_PSNR = val_avg_psnr
            best_ssim = val_avg_ssim
            torch.save(model.state_dict(), os.path.join(opt.outf, 'checkpoint', 'net_%d.pth'%epoch))

        print(opt.outf, 'Best epoch:', best_epoch, 'Best PSNR: %.4f'%best_PSNR, 'Best SSIM: %.4f'%best_ssim)
    writer.close()


if __name__ == "__main__":
    main()
