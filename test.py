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
from torchsummary import summary
from models.VDN import VDN

os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="PDNet")
parser.add_argument('--archi', type=str, default="dncnn", help='use DnCNN as reference?')
parser.add_argument('--resume', type=int, default=1, help='train or test?')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--device_ids", type=int, default=1, help="move to GPU")
parser.add_argument("--numOfLayers", type=int, default=20, help="Number of total layers")

parser.add_argument("--pretrained_path", type=str, default="", help='path of pretrained checkpoints')
parser.add_argument("--pretrained_model", type=str, default="", help='path of pretrained checkpoints')
parser.add_argument("--pretrained_archi", type=str, default="", help='archi of pretrained model')

parser.add_argument("--test_dataPath", type=str, default='data/Synthetic_set/val/CBSD68', help='path of validation files to process') #data/BSDS500_new/CBSD68
parser.add_argument("--noiseLevel", type=int, default=15, help="adjustable noise level")

parser.add_argument("--outf", type=str, default="logs/try", help='path of log files')
parser.add_argument("--model", type=str, default="", help='path of pretrained checkpoints')
# parser.add_argument('--crop', default=False, type=bool, help='crop patches?')
parser.add_argument('--cropSize', default=64, type=int,  help='crop patches? training images crop size')
parser.add_argument('--real', default=0, type=bool, help='Real Dataset?')
parser.add_argument('--seed', default=0, type= int, help='seed of all random')
# parser.add_argument('--randomCount', default=1, type= int, help='the number of patches got from each patch')
# parser.add_argument('--augment', default=True, type= bool, help='whether to apply data augmentation to it')
# parser.add_argument('--grad_weight', default=0.1, type= float, help='weight of gradient loss')
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
    VDataset = Dataset_Grad(opt.test_dataPath, randomCount=1, augment=0, cropPatch=0,
                             cropSize=opt.cropSize, real=0, noiseLevel=opt.noiseLevel)
    loaderVal = DataLoader(dataset=VDataset, num_workers=4, batch_size=opt.batchSize, shuffle=False)

    end = time.time()
    print (round(end - start, 7))    
    print("# of training samples: %d\n\n" % int(len(loaderVal)))
    
    # Build model
    if opt.archi == 'dncnn':
        net = DnCNN(channels=3, num_of_layers=opt.numOfLayers)

    if opt.archi == 'dncnn_sk':
        net = DnCNN_sk(channels=3, num_of_layers=opt.numOfLayers)

    if opt.archi == 'dncnn_sk_conGradient':
        net = DnCNN_sk_conGradient(channels=3, num_of_layers=opt.numOfLayers)

    if opt.archi == 'dncnn_sk_conGradient_before':
        net = DnCNN_sk_conGradient_before(channels=3, num_of_layers=opt.numOfLayers)

    if opt.archi == 'dncnn_sk_conGradient_onimage':
        net = DnCNN_conGradient_image(channels=3, num_of_layers=opt.numOfLayers)

    if opt.archi == 'Resnet_sk':
        net = Resnet_sk(channels=3)

    if opt.archi == 'Resnet_sk_conGradient_before':
        net = Res_conGradient_before(channels=3)

    if opt.archi == 'Resnet_sk_conGradient_onimage':
        net = Res_conGradient_onimage(channels=3)

    if opt.archi == 'Resnet_sk_conGradient':
        net = Res_conGradient(channels=3)

    if opt.archi == 'VDN':
        checkpoint = torch.load('./model_state/model_state_SIDD')
        net = VDN(3, dep_U=4, wf=64)


    # Move to GPU
    device_ids = range(opt.device_ids)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    print('archi', opt.archi)
    # Calculate the model size
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print('params:', params)

    if opt.resume == 1 :
    #     model.load_state_dict(torch.load(os.path.join(opt.outf, opt.model)))
        model.load_state_dict(checkpoint)

    model.eval()
    # validation
    val_avg_loss = 0
    val_input_psnr = 0
    val_avg_psnr = 0
    val_avg_ssim = 0
    avg_smooth_psnr = 0
    avg_texture_psnr = 0

    avg_time = 0
    with torch.no_grad():
        for i, data in enumerate(loaderVal, 0):
            imgClear, imgNoisy = data[0], data[1]  # Dataset_BSDS500_4
            imgClear, imgNoisy = imgClear.cuda(), imgNoisy.cuda()
            # imgDiff = imgNoisy - imgClear

            # add mask
            # mask, complement_mask = generate_mask(imgClear, eps=0.5) # 0.5, 1, 1.5
            # mask, complement_mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda(), torch.from_numpy(complement_mask).type(torch.FloatTensor).cuda()

            if (opt.archi == 'dncnn_sk_conGradient') or (opt.archi == 'dncnn_sk_conGradient_onimage') or (opt.archi == 'dncnn_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_onimage') or (opt.archi == 'Resnet_sk_conGradient'):
                # grad = img_gradient_total(imgClear)
                imgDenoised = get_intermediate_result(imgNoisy, opt.device_ids, opt.pretrained_archi,
                                                      opt.pretrained_path, opt.pretrained_model) # Dataset_BSDS500_4
                grad = img_gradient_total(imgDenoised)
                t0 = time.time()
                outRes = model(imgNoisy, grad)
                t1 = time.time()-t0
            else:
                t0 = time.time()
                # outRes = model(imgNoisy)
                outRes = net(imgNoisy, 'test')
                t1 = time.time()-t0

            avg_time += t1

            imgResult = torch.clamp(outRes, 0., 1.) # Clean image
            # imgResult = torch.clamp((imgNoisy-outRes), 0., 1.) # Noise
            psnrInput = batchPSNR(imgNoisy, imgClear, 1.)
            psnrVal = batchPSNR(imgResult, imgClear, 1.)
            score_ssim_val = batchSSIM(imgClear, imgResult, win_size=3, multichannel=True)

            val_input_psnr += psnrInput
            val_avg_psnr += psnrVal
            val_avg_ssim += score_ssim_val
            print("psnrInput: %.4f psnrVal: %.4f SSIM: %.4f" %
                  (psnrInput, psnrVal, score_ssim_val))

            # # different regions
            # # print('imgResult', imgResult)
            # # print('masked', imgResult*mask)
            # # print('original_masked', imgClear*mask)
            # smooth_PSNR = batchPSNR(imgResult*mask, imgClear*mask, 1.)
            # # print('imgResult', imgResult)
            #
            # texture_PSNR = batchPSNR(imgResult*complement_mask, imgClear*complement_mask, 1.)
            # # print('other_result', imgResult*complement_mask)
            # # print('other_clean', imgClear*complement_mask)
            # avg_smooth_psnr += smooth_PSNR
            # avg_texture_psnr += texture_PSNR
            # print(i, 'smooth_PSNR', smooth_PSNR, 'texture_PSNR', texture_PSNR)
            #
            # tmp = torch.clamp((mask+0.5), 0., 1.)
            # masked_denoised = imgResult*tmp
            # tmp_1 = torch.clamp((complement_mask + 0.5), 0., 1.)
            # Complementary_masked_denoised = imgResult*tmp_1
            # torchvision.utils.save_image(imgClear, os.path.join(opt.outf, 'clean.PNG'))
            # torchvision.utils.save_image(imgResult, os.path.join(opt.outf, 'imgResult.PNG'))
            # torchvision.utils.save_image(masked_denoised, os.path.join(opt.outf, 'masked_denoised.PNG'))
            # torchvision.utils.save_image(Complementary_masked_denoised, os.path.join(opt.outf, 'Complementary_masked_denoised.PNG'))
            # assert 0

        # print('avg_smooth_psnr', avg_smooth_psnr, 'avg_texture_psnr', avg_texture_psnr)
        avg_time /= len(loaderVal)
        print('Avg Time: %.5f'%avg_time)
        val_avg_loss /= len(loaderVal)
        val_avg_psnr /= len(loaderVal)
        val_avg_ssim /= len(loaderVal)
        avg_smooth_psnr /= len(loaderVal)
        avg_texture_psnr /= len(loaderVal)
        print("Avg_PSNRTr: %.4f SSIM: %.4f Avg_Smooth_PSNR: %.4f Avg_Texture_PSNR: %.4f " %
              (val_avg_psnr, val_avg_ssim, avg_smooth_psnr, avg_texture_psnr))



if __name__ == "__main__":
    main()
