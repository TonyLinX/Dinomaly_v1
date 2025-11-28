import os
import torch
from dataset import get_data_transforms
from GNL.GNL_resnet_TTA import  wide_resnet50_2
from de_resnet import  de_wide_resnet50_2
from dataset import MVTecDataset, MVTecDatasetOOD
from test import  evaluation_ATTA
from os import listdir
from PIL import Image

def evaluation_ATTA(encoder, bn, decoder, dataloader,device, type_of_test, img_size, lamda=0.5, dataset_name='mnist', _class_=None):
    bn.eval()
    decoder.eval()
    gt_list_sp = []
    pr_list_sp = []

    if dataset_name == 'mnist':
        link_to_normal_sample = '/home/cttri/anomaly/data/MNIST/MNIST_grey/training/' + str(_class_) #update the link here
        filenames = [f for f in listdir(link_to_normal_sample)]
        filenames.sort()
        link_to_normal_sample = '/home/tri/data/MNIST/MNIST_grey/training/' + str(_class_) + '/' + filenames[0] #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")

    if dataset_name == 'mvtecad2':
        link_to_normal_sample = '/DATA2/tonylinx/dev/ADShift/data/mvtec_ad_2/mvtec_ad_2/' + _class_ + '/train/good/000_regular.png' #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")
        
    if dataset_name == 'mvtec':
        link_to_normal_sample = '/DATA2/tonylinx/dev/ADShift/data/mvtec/' + _class_ + '/train/good/000.png' #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")

    if dataset_name == 'PACS':
        labels_dict = {
            0: 'dog',
            1: 'elephant',
            2: 'giraffe',
            3: 'guitar',
            4: 'horse',
            5: 'house',
            6: 'person'
        }
        link_to_normal_sample = '/home/cttri/anomaly/PACS/train/photo/' + labels_dict[_class_] #update the link here
        filenames = [f for f in listdir(link_to_normal_sample)]
        filenames.sort()
        link_to_normal_sample = '/home/cttri/anomaly/PACS/train/photo/' + labels_dict[_class_] + '/' + filenames[0] #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")

    if dataset_name != 'mnist':
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)
        ])
    else:
        trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    normal_image = trans(normal_image)
    normal_image = torch.unsqueeze(normal_image, 0)



    with torch.no_grad():
        for sample in dataloader:
            img, label = sample[0], sample[1]

            if dataset_name not in ('mvtec', 'mvtec_ood', 'mvtecad2'):
                if int(label) != _class_:
                    label = 1
                else:
                    label = 0
            else:
                label = int(torch.sum(label) != 0)


            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)

            normal_image = normal_image.to(device)
            img = img.to(device)
            inputs = encoder(img, normal_image, type_of_test, lamda=lamda)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.append(int(label))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_sp

def test_mvtec(_class_):
    # 鎖定使用 GPU 0（若無 GPU 則退回 CPU）
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Class: ', _class_)
    data_transform, gt_transform = get_data_transforms(256, 256)

    #load data
    test_path_id = './data/mvtec/' + _class_ #update here
    test_path_brightness = './data/mvtec_brightness/' + _class_ #update here
    test_path_constrast = './data/mvtec_contrast/' + _class_ #update here
    test_path_defocus_blur = './data/mvtec_defocus_blur/' + _class_ #update here
    test_path_gaussian_noise = './data/mvtec_gaussian_noise/' + _class_ #update here
    ckp_path = './checkpoints/' + 'mvtec_DINL_' + str(_class_) + '_19.pth'
    test_data_id = MVTecDataset(root=test_path_id, transform=data_transform, gt_transform=gt_transform,
                             phase="test")
    test_data_brightness = MVTecDatasetOOD(root=test_path_brightness, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_constrast = MVTecDatasetOOD(root=test_path_constrast, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_defocus_blur = MVTecDatasetOOD(root=test_path_defocus_blur, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_gaussian_noise = MVTecDatasetOOD(root=test_path_gaussian_noise, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)

    test_dataloader_id = torch.utils.data.DataLoader(test_data_id, batch_size=1, shuffle=False)
    test_dataloader_brightness = torch.utils.data.DataLoader(test_data_brightness, batch_size=1, shuffle=False)
    test_dataloader_constrast = torch.utils.data.DataLoader(test_data_constrast, batch_size=1, shuffle=False)
    test_dataloader_defocus_blur = torch.utils.data.DataLoader(test_data_defocus_blur, batch_size=1, shuffle=False)
    test_dataloader_gaussian_noise = torch.utils.data.DataLoader(test_data_gaussian_noise, batch_size=1, shuffle=False)

    #load model
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    #load checkpoint（確保載入到指定 device）
    ckp = torch.load(ckp_path, map_location=device)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    lamda = 0.5

    list_results = []
    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_id, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of ID data{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_brightness, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of brightness data{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_constrast, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of contrast data{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_defocus_blur, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of defocus blur data{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_gaussian_noise, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of Gaussian noise data{:.4f}'.format(auroc_sp))

    print(list_results)

    return

item_list = ['carpet', 'leather', 'grid', 'tile', 'wood', 'bottle', 'hazelnut', 'cable', 'capsule',
             'pill', 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper']

for i in item_list:
    test_mvtec(i)
    print('===============================================')
    print('')
    print('')

