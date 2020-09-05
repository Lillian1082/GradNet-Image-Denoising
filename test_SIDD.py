import argparse
from models.models import *
from utilsData import *
from utils import *
import scipy.io as sio
import h5py
import time

os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="Denoising")
parser.add_argument('--archi', type=str, default="dncnn_sk_conGradient_before", help='use DnCNN as reference?')
# parser.add_argument('--noc', type=int, default=0, help='Add noise on images(noi) or crops(noc)?')
parser.add_argument("--device_ids", type=int, default=1, help="move to GPU")
parser.add_argument("--numOfLayers", type=int, default=20, help="Number of total layers")
parser.add_argument("--dataPath", type=str, default="data/dnd", help='path of files to process') # Real_set
parser.add_argument("--outf", type=str, default="logs/Real_Resnet_sk_conGradient_before_3", help='path of log files')
parser.add_argument("--pretrained_path", type=str, default="logs/Real_dncnn_sk", help='path of pretrained checkpoints')
parser.add_argument("--pretrained_model", type=str, default="checkpoint/net_21.pth", help='path of pretrained checkpoints')
parser.add_argument("--pretrained_archi", type=str, default="dncnn_sk", help='archi of pretrained model')
parser.add_argument("--model", type=str, default="", help='path of pretrained checkpoints')
parser.add_argument('--seed', default=0, type= int, help='seed of all random')
opt = parser.parse_args()

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)

def SIDD_test(model, opt, outfile):
    model.eval()
    print(os.path.join(opt.outf, outfile)) #SIDD_test_output
    if not os.path.exists(os.path.join(opt.outf, outfile)): #test_output
        os.mkdir(os.path.join(opt.outf, outfile))
        print('Done')

    print('model loaded\n')
    # load info
    files = scipy.io.loadmat(os.path.join(opt.dataPath, 'BenchmarkNoisyBlocksSrgb.mat'))
    imgArray = files['BenchmarkNoisyBlocksSrgb']
    nImages = 40
    nBlocks = imgArray.shape[1]
    DenoisedBlocksSrgb = np.empty_like(imgArray)
    # process data
    for i in range(nImages):
        Inoisy = normalize(np.float32(imgArray[i]))
        Inoisy = torch.from_numpy(np.transpose(Inoisy, (0, 3, 1, 2))).type(torch.FloatTensor).cuda()

        with torch.no_grad():  # this can save much memory
            if (opt.archi == 'dncnn_sk_conGradient') or (opt.archi == 'dncnn_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before2'):
                print('archi', opt.archi)
                imgDenoised = get_intermediate_result(Inoisy, opt.device_ids, opt.pretrained_archi, opt.pretrained_path, opt.pretrained_model)
                grad = img_gradient_total(imgDenoised)
                outRes = model(Inoisy, grad)
            else:
                outRes = model(Inoisy)
            # save denoised data
            outRes=torch.clamp(outRes, 0., 1.) # output image
            # outRes = torch.clamp(Inoisy - outRes, 0., 1.)  # output noise
            Idenoised_crop= outRes.cpu().detach().numpy()*255
            # Idenoised_crop = Idenoised_crop.astype(np.uint8)
            Idenoised_crop = np.transpose(Idenoised_crop, (0, 2, 3, 1))
            # Idenoised_crop = np.expand_dims(Idenoised_crop, axis=0)
            DenoisedBlocksSrgb[i] = Idenoised_crop

    # save_file = os.path.join(opt.outf, 'SIDD_test_output','SubmitSrgb.mat') # SIDD_test_output
    # sio.savemat(save_file, {'DenoisedBlocksSrgb': DenoisedBlocksSrgb, 'TimeMPSrgb' : 0.0})
            for k in range(nBlocks):
                # a = Idenoised_crop[k]
            #     save_file_noisy = os.path.join(opt.outf, 'SIDD_test_output', '%04d_%02d_noisy.PNG' % (i + 1, k + 1))
            #     save_file = os.path.join(opt.outf, outfile,'%04d_%02d.PNG'%(i+1,k+1))
            #     noisy_image = imgArray[i, k]
            #     cv2.imwrite(save_file, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))

                save_file = os.path.join(opt.outf, outfile, '%04d_%02d.PNG' % (i + 1, k + 1))
                cv2.imwrite(save_file, cv2.cvtColor(Idenoised_crop[k], cv2.COLOR_RGB2BGR))
                #np.save(save_file, Idenoised_crop[k])
            # #     sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop[k]})
            #     print('%s crop %d/%d' % (save_file, k+1, Idenoised_crop.shape[0]))
                print('[%d/%d] is done\n' % (i+1, 40))


def DND_test(model, opt, outfile):
    model.eval()
    print(os.path.join(opt.outf, outfile)) #SIDD_test_output
    if not os.path.exists(os.path.join(opt.outf, outfile)): #test_output
        os.mkdir(os.path.join(opt.outf, outfile))
        print('Done')
    print('model loaded\n')

    noisy_files = glob.glob(os.path.join(opt.dataPath, '*.*'))
    noisy_files.sort()

    # process data
    for i in range(len(noisy_files)):
        imgNoisy = np.load(noisy_files[i])
        imgNoisy = torch.from_numpy(np.moveaxis(imgNoisy, -1, 0)).type(torch.FloatTensor).cuda()
        if len(imgNoisy.shape) == 3:
            imgNoisy = torch.unsqueeze(imgNoisy, dim=0)
        with torch.no_grad():  # this can save much memory
            if (opt.archi == 'dncnn_sk_conGradient') or (opt.archi == 'dncnn_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before2'):
                print('archi', opt.archi)
                imgDenoised = get_intermediate_result(imgNoisy, opt.device_ids, opt.pretrained_archi, opt.pretrained_path, opt.pretrained_model)
                grad = img_gradient_total(imgDenoised)
                t0 = time.time()
                outRes = model(imgNoisy, grad)
                t1 = time.time() - t0
            else:
                t0 = time.time()
                outRes = model(imgNoisy)
                t1 = time.time() - t0
            imgResult = torch.clamp(outRes, 0., 1.)
            imgResult = imgResult.cpu().detach().squeeze().numpy().astype(np.float32)
            imgResult = np.transpose(imgResult, (1, 2, 0)) # * 255

            # file_name = os.path.join(opt.outf, outfile, noisy_files[i].split('/')[-1][:-4])+'.png'
            # cv2.imwrite(file_name, cv2.cvtColor(imgResult*255, cv2.COLOR_RGB2BGR))
            file_name = os.path.join(opt.outf, outfile, noisy_files[i].split('/')[-1][:-4]) + '.npy'
            np.save(file_name, imgResult)
        print('[%d/%d] is done\n' % (i+1, len(noisy_files)))


def RNI_test(model, opt, outfile):
    model.eval()
    print(os.path.join(opt.outf, outfile)) #SIDD_test_output
    if not os.path.exists(os.path.join(opt.outf, outfile)): #test_output
        os.mkdir(os.path.join(opt.outf, outfile))
        print('Done')
    print('model loaded\n')

    noisy_files = glob.glob(os.path.join(opt.dataPath, '*.*'))
    noisy_files.sort()

    # process data
    for i in range(len(noisy_files)):
        imgNoisy = normalize(cv2.imread(noisy_files[i]))
        imgNoisy = torch.from_numpy(np.moveaxis(imgNoisy, -1, 0)).type(torch.FloatTensor).cuda()
        if len(imgNoisy.shape) == 3:
            imgNoisy = torch.unsqueeze(imgNoisy, dim=0)
        with torch.no_grad():  # this can save much memory
            if (opt.archi == 'dncnn_sk_conGradient') or (opt.archi == 'dncnn_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before') or (opt.archi == 'Resnet_sk_conGradient_before2'):
                print('archi', opt.archi)
                imgDenoised = get_intermediate_result(imgNoisy, opt.device_ids, opt.pretrained_archi, opt.pretrained_path, opt.pretrained_model)
                grad = img_gradient_total(imgDenoised)
                outRes = model(imgNoisy, grad)
            else:
                outRes = model(imgNoisy)
            imgResult = torch.clamp(outRes, 0., 1.)
            imgResult = imgResult.cpu().detach().squeeze().numpy().astype(np.float32)
            imgResult = np.transpose(imgResult, (1, 2, 0)) # * 255

            file_name = os.path.join(opt.outf, outfile, noisy_files[i].split('/')[-1][:-4])+'.png'
            cv2.imwrite(file_name, imgResult*255) #cv2.cvtColor(imgResult*255, cv2.COLOR_RGB2BGR)
            # file_name = os.path.join(opt.outf, outfile, noisy_files[i].split('/')[-1][:-4]) + '.npy'
            # np.save(file_name, imgResult)
        print('[%d/%d] is done\n' % (i+1, len(noisy_files)))

def SIDD_CROP_test(model):
    model.eval()
    print(os.path.join(opt.outf, 'SIDD_test_output')) #SIDD_test_output
    if not os.path.exists(os.path.join(opt.outf, 'SIDD_test_output')): #test_output
        os.mkdir(os.path.join(opt.outf, 'SIDD_test_output'))
        print('Done')

    print('model loaded\n')
    # load info
    files = scipy.io.loadmat(os.path.join(opt.dataPath, 'BenchmarkNoisyBlocksSrgb.mat'))
    imgArray = files['BenchmarkNoisyBlocksSrgb']
    nImages = 40
    nBlocks = imgArray.shape[1]
    DenoisedBlocksSrgb = np.empty_like(imgArray)
    # process data
    for i in range(nImages):
        Inoisy = normalize(np.float32(imgArray[i]))
        cv2.imwrite('data/SIDD/test', cv2.cvtColor(Inoisy, cv2.COLOR_RGB2BGR))
    #     Inoisy = torch.from_numpy(np.transpose(Inoisy, (0, 3, 1, 2))).type(torch.FloatTensor).cuda()
    #
    #     with torch.no_grad():  # this can save much memory
    #         if (opt.archi == 'dncnn_sk_conGradient') or (opt.archi == 'dncnn_sk_conGradient_before'):
    #             print('archi', opt.archi)
    #             imgDenoised = get_intermediate_result(Inoisy, opt)
    #             grad = img_gradient_total(imgDenoised)
    #             outRes = model(Inoisy, grad)
    #         else:
    #             outRes = model(Inoisy)
    #         # save denoised data
    #         outRes=torch.clamp(outRes, 0., 1.)
    #         Idenoised_crop= outRes.cpu().detach().numpy()*255
    #         # Idenoised_crop = Idenoised_crop.astype(np.uint8)
    #         Idenoised_crop = np.transpose(Idenoised_crop, (0, 2, 3, 1))
    #         # Idenoised_crop = np.expand_dims(Idenoised_crop, axis=0)
    #         DenoisedBlocksSrgb[i] = Idenoised_crop
    #
    # save_file = os.path.join(opt.outf, 'SIDD_test_output','SubmitSrgb.mat') # SIDD_test_output
    # sio.savemat(save_file, {'DenoisedBlocksSrgb': DenoisedBlocksSrgb, 'TimeMPSrgb' : 0.0})
            # for k in range(nBlocks):
            #     a = Idenoised_crop[k]
            #     save_file_noisy = os.path.join(opt.outf, 'SIDD_test_output', '%04d_%02d_noisy.PNG' % (i + 1, k + 1))
            #     save_file = os.path.join(opt.outf, 'SIDD_test_output','%04d_%02d.PNG'%(i+1,k+1))
            #     noisy_image = imgArray[i, k]
            #     cv2.imwrite(save_file, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
            #     cv2.imwrite(save_file, cv2.cvtColor(Idenoised_crop[k], cv2.COLOR_RGB2BGR))
            # #     sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop[k]})
            #     print('%s crop %d/%d' % (save_file, k+1, Idenoised_crop.shape[0]))
            #     print('[%d/%d] is done\n' % (i+1, 40))

def main():
    # Load dataset
    print('Loading dataset ...\n')

    if opt.archi == 'dncnn':
        net = DnCNN(channels=3, num_of_layers=opt.numOfLayers)
    if opt.archi == 'dncnn_sk':
        net = DnCNN_sk(channels=3, num_of_layers=opt.numOfLayers)
    if opt.archi == 'dncnn_sk_conGradient':
        net = DnCNN_sk_conGradient(channels=3, num_of_layers=opt.numOfLayers)
    if opt.archi == 'dncnn_sk_conGradient_before':
        net = DnCNN_sk_conGradient_before(channels=3, num_of_layers=opt.numOfLayers)
    if opt.archi == 'Resnet_sk_conGradient_before':
        net = Res_conGradient_before(channels=3)
    if opt.archi == 'Resnet_sk':
        net = Resnet_sk(channels=3)

    device_ids = range(opt.device_ids)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.outf, opt.model)))

    SIDD_test(model, opt, 'SIDD_val_output_img_'+opt.outf.split('/')[1])
    # DND_test(model, opt, 'DND_test_output_npy_'+opt.outf.split('/')[1])
    # RNI_test(model, opt, 'RNI15_test_output_img_' + opt.outf.split('/')[1])
    # RNI_test(model, opt, 'SIDD_val_output_img_' + opt.outf.split('/')[1])



if __name__ == "__main__":
    main()