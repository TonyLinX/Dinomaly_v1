import argparse
import torch
import os
from datetime import datetime
from torchvision.datasets import ImageFolder
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataset import AugMixDatasetMVTec
from seed_utils import set_seed, seed_worker, build_generator


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def loss_fucntion(a, b):
    # mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        # print(a[item].shape)
        # print(b[item].shape)
        # loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss

def loss_fucntion_last(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # for item in range(len(a)):
    #     # print(a[item].shape)
    #     # print(b[item].shape)
    #     # loss += 0.1*mse_loss(a[item], b[item])
    item = 0
    loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                    b[item].view(b[item].shape[0], -1)))
    return loss



def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        # loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss


def train(_class_, device, seed, num_workers):
    print(_class_)
    epochs = 20
    learning_rate = 0.005
    batch_size = 16
    image_size = 256

    print(device)

    # 準備日誌檔案
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"mvtec_DINL_{_class_}_seed{seed}_" + datetime.now().strftime('%Y%m%d-%H%M%S') + ".log")
    with open(log_path, 'w') as f:
        f.write(f"class={_class_}, device={device}, seed={seed}\n")
        f.write(f"epochs={epochs}, lr={learning_rate}, batch_size={batch_size}, image_size={image_size}\n")

    # 準備 checkpoint 目錄
    ckpt_dir = './checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)


    resize_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
    ])


    train_path = './data/mvtec/' + _class_ + '/train' #update here
    train_data = ImageFolder(root=train_path, transform=resize_transform)
    train_data = AugMixDatasetMVTec(train_data, preprocess)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=build_generator(seed),
        pin_memory=(device.type == 'cuda'))

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))



    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        # 累積各項 loss 以計算每個 epoch 的平均
        loss_total_sum = 0.0
        loss_normal_sum = 0.0
        loss_bn_sum = 0.0
        loss_last_sum = 0.0
        batch_count = 0
        for normal, augmix_img, gray_img in train_dataloader:
            normal = normal.to(device)
            inputs_normal = encoder(normal)
            bn_normal = bn(inputs_normal)
            outputs_normal = decoder(bn_normal)  


            augmix_img = augmix_img.to(device)
            inputs_augmix = encoder(augmix_img)
            bn_augmix = bn(inputs_augmix)
            outputs_augmix = decoder(bn_augmix)

            gray_img = gray_img.to(device)
            inputs_gray = encoder(gray_img)
            bn_gray = bn(inputs_gray)

            loss_bn = loss_fucntion([bn_normal], [bn_augmix]) + loss_fucntion([bn_normal], [bn_gray])
            outputs_gray = decoder(bn_gray)

            loss_last = loss_fucntion_last(outputs_normal, outputs_augmix) + loss_fucntion_last(outputs_normal, outputs_gray)

            loss_normal = loss_fucntion(inputs_normal, outputs_normal)
            loss = loss_normal*0.9 + loss_bn*0.05 + loss_last*0.05

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # 累積 batch loss
            loss_total_sum += loss.item()
            loss_normal_sum += loss_normal.item()
            loss_bn_sum += loss_bn.item()
            loss_last_sum += loss_last.item()
            batch_count += 1

        # 計算 epoch 平均 loss 並寫入日誌
        if batch_count > 0:
            avg_loss = loss_total_sum / batch_count
            avg_loss_normal = loss_normal_sum / batch_count
            avg_loss_bn = loss_bn_sum / batch_count
            avg_loss_last = loss_last_sum / batch_count
        else:
            avg_loss = avg_loss_normal = avg_loss_bn = avg_loss_last = 0.0

        msg = (f"Epoch [{epoch+1}/{epochs}] avg_loss={avg_loss:.6f} "
               f"normal={avg_loss_normal:.6f} bn={avg_loss_bn:.6f} last={avg_loss_last:.6f}")
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + "\n")

        if (epoch + 1) % 20 == 0 :
            ckp_path = os.path.join(ckpt_dir, 'mvtec_DINL_' + str(_class_) + '_' + str(epoch) + '.pth')
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
            print(f"Saved checkpoint: {ckp_path}")
            with open(log_path, 'a') as f:
                f.write(f"Saved checkpoint: {ckp_path}\n")
        



    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DINL on MVTec')
    parser.add_argument('--gpu', type=int, default=0, help='要使用的 GPU 編號；設為 -1 則用 CPU')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker count')
    args = parser.parse_args()

    if args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    set_seed(args.seed)

    item_list = ['carpet', 'leather', 'grid', 'tile', 'wood', 'bottle', 'hazelnut', 'cable', 'capsule',
                  'pill', 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper']
    for i in item_list:
        train(i, device, args.seed, args.num_workers)
