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
# from loss import *
from math import log10
from utilsData import *
from utils import *

os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="PDNet")
parser.add_argument('--archi', type=str, default="dncnn_sk_conGradient_before", help='use DnCNN as reference?')
parser.add_argument('--resume', type=int, default=1, help='train or finetune?')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--device_ids", type=int, default=1, help="move to GPU")
parser.add_argument("--numOfLayers", type=int, default=20, help="Number of total layers")
parser.add_argument("--dataPath", type=str, default='data/Kodak_CROP', help='path of files to process')

parser.add_argument("--pretrained_path", type=str, default="", help='path of pretrained checkpoints')
parser.add_argument("--pretrained_model", type=str, default="", help='path of pretrained checkpoints')
parser.add_argument("--pretrained_archi", type=str, default="", help='archi of pretrained model')

parser.add_argument("--noiseLevel", type=int, default=50, help="adjustable noise level")
parser.add_argument("--outf", type=str, default="logs/BSDS500_dncnn_sk_conGradient_before_image_S15_bm3d_2", help='path of log files')
parser.add_argument("--model", type=str, default="", help='path of pretrained checkpoints')
parser.add_argument('--seed', default=0, type= int, help='seed of all random')
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
    gt_files = glob.glob(os.path.join(opt.dataPath, 'GT', '*.*')) #image
    gt_files.sort()
    # noisy_files = glob.glob(os.path.join(opt.dataPath, 'Noisy_S%d'%opt.noiseLevel, '*.*')) #_
    # noisy_files.sort()
    # denoised_files = glob.glob(os.path.join(opt.dataPath, 'Denoised_S%d'%opt.noiseLevel, '*.*'))
    # denoised_files.sort()

    end = time.time()
    print (round(end - start, 7))
    
    # Build model
    if opt.archi == 'dncnn':
        net = DnCNN(channels=3, num_of_layers=opt.numOfLayers)

    if opt.archi == 'dncnn_sk':
        net = DnCNN_sk(channels=3, num_of_layers=opt.numOfLayers)

    if opt.archi == 'dncnn_sk_conGradient_before':
        net = DnCNN_sk_conGradient_before(channels=3, num_of_layers=opt.numOfLayers)

    if opt.archi == 'Resnet_sk_conGradient_before':
        net = Res_conGradient_before(channels=3)

    net.apply(weights_init_kaiming)

    # Move to GPU
    device_ids = range(opt.device_ids)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    if opt.resume == 1 :
        model.load_state_dict(torch.load(os.path.join(opt.outf, opt.model)))

    if not os.path.exists(os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_output')):
        os.mkdir(os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_output'))
    if not os.path.exists(os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_grad')):
        os.mkdir(os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_grad'))

    # writer = SummaryWriter(opt.outf)

    avg_psnr_in = 0
    avg_psnr_out = 0
    avg_ssim = 0
    avg_time = 0
    with torch.no_grad():
        for i in range(len(gt_files)):
            # training step
            model.eval()

            # extract data and make predictions
            if gt_files[i].split('/')[-1][-4:] == '.npy':
                imgClear = np.load(gt_files[i])
                # imgNoisy = np.load(noisy_files[i])
                # imgDenoised = np.load(denoised_files[i])
            else :
                imgClear = normalize(cv2.imread(gt_files[i])) # read .png/.jpg/.tif
                # imgNoisy = normalize(cv2.imread(noisy_files[i]))  # read .png/.jpg/.tif
                # imgDenoised = normalize(cv2.imread(denoised_files[i]))  # read .png/.jpg/.tif
            imgNoisy = addNoise(imgClear, sigma=25)
            imgClear = torch.from_numpy(np.moveaxis(imgClear, -1, 0)).type(torch.FloatTensor).cuda()
            imgNoisy = torch.from_numpy(np.moveaxis(imgNoisy, -1, 0)).type(torch.FloatTensor).cuda()
            # imgDenoised = torch.from_numpy(np.moveaxis(imgDenoised, -1, 0)).type(torch.FloatTensor).cuda()

            # Calculate PSNR
            # psnrIn = batchPSNR(imgDenoised, imgClear, 1.)
            # print(psnrIn)
            # assert 0

            if len(imgClear.shape) == 3:
                # imgClear, imgNoisy, imgDenoised = torch.unsqueeze(imgClear, dim=0), torch.unsqueeze(imgNoisy, dim=0), torch.unsqueeze(imgDenoised, dim=0)
                imgClear, imgNoisy = torch.unsqueeze(imgClear, dim=0), torch.unsqueeze(imgNoisy, dim=0)
            if (opt.archi =='dncnn_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before'):
                imgDenoised = get_intermediate_result(imgNoisy, opt.device_ids, opt.pretrained_archi, opt.pretrained_path, opt.pretrained_model)
                grad = img_gradient_total(imgDenoised)
                t0 = time.time()
                outRes = model(imgNoisy, grad)
                t1 = time.time() - t0
            else:
                t0 = time.time()
                outRes = model(imgNoisy)
                t1 = time.time() - t0

            avg_time += t1
            # results
            imgResult = torch.clamp(outRes, 0., 1.) # Output clean image
            # imgResult = torch.clamp((imgNoisy-outRes), 0., 1.) # Output noise
            psnrIn = batchPSNR(imgNoisy, imgClear, 1.)
            psnrOut = batchPSNR(imgResult, imgClear, 1.)
            # psnrPreDenoised = batchPSNR(imgDenoised, imgClear, 1.)
            score_ssim = batchSSIM(imgClear, imgResult, win_size=3, multichannel=True)
            avg_psnr_in += psnrIn
            avg_psnr_out += psnrOut
            avg_ssim += score_ssim
            print(gt_files[i], "PSNRInput: %.4f PSNROutput: %.4f SSIM: %.4f" % (
            psnrIn, psnrOut, score_ssim))
            # # save some images
            # noise = imgNoisy-imgClear
            # noise = (noise - noise.min()) / (noise.max() - noise.min())
            # removed_noise = imgNoisy-outRes # output clean image
            final_denoised = imgResult
            # removed_noise = (removed_noise-removed_noise.min())/(removed_noise.max()-removed_noise.min())
            #
            # torchvision.utils.save_image(imgClear, os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_output', gt_files[i].split('/')[4].split('.')[0]+'_clean.PNG'))
            # torchvision.utils.save_image(imgNoisy, os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_output', gt_files[i].split('/')[-1].split('.')[0]+'_noisy.PNG'))
            # torchvision.utils.save_image(imgDenoised, os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_output',
            #                                                     gt_files[i].split('/')[4].split('.')[0]+'_Denoised.PNG'))
            # torchvision.utils.save_image(noise, os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_output',
            #                                                     gt_files[i].split('/')[4].split('.')[0]+'_noise.PNG'))
            # torchvision.utils.save_image(removed_noise, os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_output_2',
            #                                                     gt_files[i].split('/')[4].split('.')[0]+'_removed_noise.PNG'))
            # torchvision.utils.save_image(final_denoised, os.path.join(opt.outf, opt.dataPath.split('/')[-1]+'_test_output',
            #                                                     gt_files[i].split('/')[-1].split('.')[0]+'_denoised.PNG'))

            # # save some numbers
            # is_exist = os.path.isfile(os.path.join(opt.outf, 'test_results.csv'))
            # if not is_exist:
            #     with open(os.path.join(opt.outf, 'test_results.csv'), 'a+') as writeFile:
            #         fieldnames = ['FileName', 'PSNR_Input', 'PSNR_Output', 'PSNR_PreDenoised', 'SSIM']
            #         file_writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',')
            #         file_writer.writeheader()
            #         data = [{'FileName': gt_files[i], 'PSNR_Input': psnrIn, 'PSNR_Output': psnrOut, 'PSNR_PreDenoised': psnrPreDenoised, 'SSIM': score_ssim}]
            #         file_writer.writerows(data)
            # else:
            #     with open(os.path.join(opt.outf, 'test_results.csv'), 'a+') as writeFile:
            #         fieldnames = ['FileName', 'PSNR_Input', 'PSNR_Output', 'PSNR_PreDenoised', 'SSIM']
            #         file_writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',')
            #         data = [{'FileName': gt_files[i], 'PSNR_Input': psnrIn, 'PSNR_Output': psnrOut, 'PSNR_PreDenoised': psnrPreDenoised, 'SSIM': score_ssim}]
            #         file_writer.writerows(data)
            # writeFile.close()

        avg_time /= len(gt_files)
        print('Avg Time: %.5f seconds'%avg_time)

        avg_psnr_in /= len(gt_files)
        avg_psnr_out /= len(gt_files)
        avg_ssim /= len(gt_files)
        print("Total PSNRInput: %.4f PSNROutput: %.4f SSIM: %.4f" % (avg_psnr_in, avg_psnr_out, avg_ssim))
    # writer.close()


if __name__ == "__main__":
    main()
