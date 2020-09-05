import os.path
from utils import *

class Dataset_Grad(Dataset):
    def __init__(self, path, **kwargs):
        super(Dataset_Grad, self).__init__()
        self.real = kwargs['real'] # Real or Synthetic DataSet
        self.crop = kwargs['cropPatch']
        self.cropSize = kwargs['cropSize']
        self.randomCount = kwargs['randomCount']
        self.augment = kwargs['augment']
        self.noiseLevel = kwargs['noiseLevel']
        self.gt_files = glob.glob(os.path.join(path, 'GT', '*.*'))
        self.gt_files.sort()
        if self.real:
            self.noisy_files = glob.glob(os.path.join(path, 'Noisy', '*.*'))
            # self.noisy_files = glob.glob(os.path.join(path, 'Noisy_S%d'%self.noiseLevel, '*.*'))
            self.noisy_files.sort()
        # self.noisy_path = os.path.join(path, 'Noisy_S50')
        print(len(self.gt_files))
        print('real', self.real)
        print('aug', self.augment)
        print('crop', self.crop)

    def __getitem__(self, index):
        # print(index)
        # print(self.gt_files[index // self.randomCount])
        if self.gt_files[index // self.randomCount].split('/')[-1][-4:]=='.npy':
            image = np.load(self.gt_files[index // self.randomCount])
        else:
            image = normalize(cv2.imread(self.gt_files[index // self.randomCount]))
        if self.real:
            if self.gt_files[index // self.randomCount].split('/')[-1][-4:] == '.npy':
                noisy_image = np.load(self.noisy_files[index // self.randomCount])
            else:
                noisy_image = normalize(cv2.imread(self.noisy_files[index // self.randomCount]))
        else:
            noisy_image = addNoise(image, self.noiseLevel)
        # image_name = self.gt_files[index // self.randomCount].split('/')[-1]
        # np.save(os.path.join(self.noisy_path, image_name), noisy_image)
        if self.crop:
            endw, endh = image.shape[0], image.shape[1]
            assert (endw >= self.cropSize) and (endh >= self.cropSize)
            x = np.random.randint(0, endw - self.cropSize)
            y = np.random.randint(0, endh - self.cropSize)
            image = image[x:(self.cropSize + x), y:(self.cropSize + y), :]
            noisy_image = noisy_image[x:(self.cropSize + x), y:(self.cropSize + y), :]
        if self.augment:
            def _augment(img, noisy_img):
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                rot90 = random.random() < 0.5
                if hflip:
                    img = img[:, ::-1, :]
                    noisy_img = noisy_img[:, ::-1, :]
                if vflip:
                    img = img[::-1, :, :]
                    noisy_img = noisy_img[::-1, :, :]
                if rot90:
                    img = img.transpose(1, 0, 2)
                    noisy_img = noisy_img.transpose(1, 0, 2)
                return img, noisy_img
            image, noisy_image = _augment(image, noisy_image)
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(np.copy(image))
        noisy_image = torch.from_numpy(np.copy(np.moveaxis(noisy_image, -1, 0)))
        return (image.type(torch.FloatTensor), noisy_image.type(torch.FloatTensor))

    def __len__(self):
        return len(self.gt_files) * self.randomCount

# # Multi-wavelet
# class Dataset_SIDD5(Dataset): # concatenate images
#     def __init__(self, path, **kwargs):
#         super(Dataset_SIDD5, self).__init__()
#         self.crop = kwargs['cropPatch']
#         self.cropSize = kwargs['cropSize']
#         self.randomCount = kwargs['randomCount']
#         self.augment = kwargs['augment']
#         self.gt_files = glob.glob(os.path.join(path, 'GT', '*.*')) # train_crop_256_10, train_crop_512_10
#         self.gt_files.sort()
#         self.noisy_files = glob.glob(os.path.join(path, 'Noisy', '*.*'))
#         self.noisy_files.sort()
#         print(len(self.gt_files))
#
#     def __getitem__(self, index):
#         # print(index)
#         image = normalize(cv2.imread(self.gt_files[index // self.randomCount]))
#         noisy_image = normalize(cv2.imread(self.noisy_files[index // self.randomCount]))
#         if self.crop:
#             endw, endh = image.shape[0], image.shape[1]
#             assert (endw >= self.cropSize) and (endh >= self.cropSize)
#             x = np.random.randint(0, endw - self.cropSize)
#             y = np.random.randint(0, endh - self.cropSize)
#             image = image[x:(self.cropSize + x), y:(self.cropSize + y), :]
#             noisy_image = noisy_image[x:(self.cropSize + x), y:(self.cropSize + y), :]
#         if self.augment:
#             def _augment(img, noisy_img):
#                 hflip = random.random() < 0.5
#                 vflip = random.random() < 0.5
#                 rot90 = random.random() < 0.5
#                 if hflip:
#                     img = img[:, ::-1, :]
#                     noisy_img = noisy_img[:, ::-1, :]
#                 if vflip:
#                     img = img[::-1, :, :]
#                     noisy_img = noisy_img[::-1, :, :]
#                 if rot90:
#                     img = img.transpose(1, 0, 2)
#                     noisy_img = noisy_img.transpose(1, 0, 2)
#                 return img, noisy_img
#             image, noisy_image = _augment(image, noisy_image)
#         image = np.moveaxis(image, -1, 0)
#         # image = torch.from_numpy(np.copy(image))
#
#         # cv2.imwrite('Noisy.jpg', noisy_image*255)
#         noisy = np.copy(np.moveaxis(noisy_image, -1, 0))
#         mw_noisy_image = np.copy(noisy)
#         for i in range(noisy.shape[0]):
#             coeffs2 = pywt.dwt2(noisy[i], 'bior1.3')
#             LL, (LH, HL, HH) = coeffs2
#             for j, a in enumerate([LL, LH, HL, HH]):
#                 a = cv2.resize(a, (noisy[i].shape[1], noisy[i].shape[0]), interpolation=cv2.INTER_CUBIC)
#                 # cv2.imwrite('%d_%d.jpg'%(i, j), a*255)
#                 a = np.expand_dims(a, axis=0)
#                 mw_noisy_image = np.concatenate((mw_noisy_image, a), axis=0)
#         noisy = torch.from_numpy(noisy)
#         mw_noisy_image = torch.from_numpy(mw_noisy_image)
#         return (image.type(torch.FloatTensor), noisy.type(torch.FloatTensor), mw_noisy_image.type(torch.FloatTensor))
#
#     def __len__(self):
#         return len(self.gt_files) * self.randomCount
# # Multi-wavelet2
# class Dataset_SIDD6(Dataset): # mwresnet concatenate features
#     def __init__(self, path, **kwargs):
#         super(Dataset_SIDD6, self).__init__()
#         self.crop = kwargs['cropPatch']
#         self.cropSize = kwargs['cropSize']
#         self.randomCount = kwargs['randomCount']
#         self.augment = kwargs['augment']
#         self.gt_files = glob.glob(os.path.join(path, 'GT', '*.*')) # train_crop_256_10, train_crop_512_10
#         self.gt_files.sort()
#         self.noisy_files = glob.glob(os.path.join(path, 'Noisy', '*.*'))
#         self.noisy_files.sort()
#         print(len(self.gt_files))
#
#     def __getitem__(self, index):
#         # print(index)
#         image = normalize(cv2.imread(self.gt_files[index // self.randomCount]))
#         noisy_image = normalize(cv2.imread(self.noisy_files[index // self.randomCount]))
#         if self.crop:
#             endw, endh = image.shape[0], image.shape[1]
#             assert (endw >= self.cropSize) and (endh >= self.cropSize)
#             x = np.random.randint(0, endw - self.cropSize)
#             y = np.random.randint(0, endh - self.cropSize)
#             image = image[x:(self.cropSize + x), y:(self.cropSize + y), :]
#             noisy_image = noisy_image[x:(self.cropSize + x), y:(self.cropSize + y), :]
#         if self.augment:
#             def _augment(img, noisy_img):
#                 hflip = random.random() < 0.5
#                 vflip = random.random() < 0.5
#                 rot90 = random.random() < 0.5
#                 if hflip:
#                     img = img[:, ::-1, :]
#                     noisy_img = noisy_img[:, ::-1, :]
#                 if vflip:
#                     img = img[::-1, :, :]
#                     noisy_img = noisy_img[::-1, :, :]
#                 if rot90:
#                     img = img.transpose(1, 0, 2)
#                     noisy_img = noisy_img.transpose(1, 0, 2)
#                 return img, noisy_img
#             image, noisy_image = _augment(image, noisy_image)
#         image = np.moveaxis(image, -1, 0)
#         image = torch.from_numpy(np.copy(image))
#
#         noisy = np.copy(np.moveaxis(noisy_image, -1, 0))
#         for i in range(noisy.shape[0]):
#             coeffs2 = pywt.dwt2(noisy[i], 'bior1.3')
#             LL, (LH, HL, HH) = coeffs2
#             for j, a in enumerate([LL, LH, HL, HH]):
#                 a = cv2.resize(a, (noisy[i].shape[1], noisy[i].shape[0]), interpolation=cv2.INTER_CUBIC)
#                 # cv2.imwrite('%d_%d.jpg'%(i, j), a*255)
#                 a = np.expand_dims(a, axis=0)
#                 if (i==0) and (j==0):
#                     mw_noisy_image = a
#                 else:
#                     mw_noisy_image = np.concatenate((mw_noisy_image, a), axis=0)
#         noisy = torch.from_numpy(noisy)
#         mw_noisy_image = torch.from_numpy(mw_noisy_image)
#         return (image.type(torch.FloatTensor), noisy.type(torch.FloatTensor), mw_noisy_image.type(torch.FloatTensor))
#
#     def __len__(self):
#         return len(self.gt_files) * self.randomCount



# if __name__ == '__main__':
#     main()

