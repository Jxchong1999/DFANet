from models.DFANet import *
import torch.optim as optim
import numpy as np
import os
import random
import torchvision.transforms as T


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def get_data_loader(data_path="", data_path2="live", batch_size=5, shuffle=True, drop_last=True):
    # data path 
    data = None
    live_spoof_label = None
    material_label = None

    if data_path2 == "live":
        data = np.load(data_path)
        material_label = np.ones(len(data), dtype=np.int64)
        live_spoof_label = np.ones(len(data), dtype=np.int64)

    else:
        print_data = np.load(data_path)
        replay_data = np.load(data_path2)
        data = np.concatenate((print_data, replay_data), axis=0)
        print_lab = np.zeros(len(print_data), dtype=np.int64)
        replay_lab = np.ones(len(replay_data), dtype=np.int64) * 2
        material_label = np.concatenate((print_lab, replay_lab), axis=0)
        live_spoof_label = np.zeros(len(data), dtype=np.int64)

    # dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label),
                                              torch.tensor(material_label))
    # free memory
    import gc
    del data
    gc.collect()
    # dataloader
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader


def get_inf_iterator(data_loader):
    # """Inf data iterator."""
    while True:
        for images, live_spoof_labels, material_label in data_loader:
            yield (images, live_spoof_labels, material_label)






seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True 

device_id = "cuda:0" 
results_path = "/var/mplab_share_data/jxchong/testing/fwt(i)"

# R_C_M R_O_M O_C_M R_C_O 
# replay casia Oulu MSU  WW 
dataset1 = "casia"
dataset2 = "Oulu"
dataset3 = "MSU"

batch_size = 5
log_step = 20
model_save_step = 20
model_save_epoch = 1
save_index = 0

live_path1 =    '/var/mplab_share_data/domain-generalization/' + dataset1 + '_images_live.npy'
live_path2 =    '/var/mplab_share_data/domain-generalization/' + dataset2 + '_images_live.npy'
live_path3 =    '/var/mplab_share_data/domain-generalization/' + dataset3 + '_images_live.npy'

print_path1 =   '/var/mplab_share_data/domain-generalization/' + dataset1 + '_print_images.npy'
print_path2 =   '/var/mplab_share_data/domain-generalization/' + dataset2 + '_print_images.npy'
print_path3 =   '/var/mplab_share_data/domain-generalization/' + dataset3 + '_print_images.npy'

replay_path1 =  '/var/mplab_share_data/domain-generalization/' + dataset1 + '_replay_images.npy'
replay_path2 =  '/var/mplab_share_data/domain-generalization/' + dataset2 + '_replay_images.npy'
replay_path3 =  '/var/mplab_share_data/domain-generalization/' + dataset3 + '_replay_images.npy'

Fas_Net = Ad_LDCNet().to(device_id)
criterion_ce = nn.CrossEntropyLoss().to(device_id)
criterionMSE = torch.nn.MSELoss().to(device_id)
criterion_cosine = nn.CosineSimilarity().to(device_id)

optimizer_fas =     optim.AdamW(Fas_Net.parameters(), lr=1e-4, betas=(0.5, 0.8))#, eps=1e-08, weight_decay=1e-6, amsgrad=False)
optimizer_fwt =     optim.AdamW(Fas_Net.parameters(), lr=1e-4, betas=(0.5, 0.8))#, eps=1e-08, weight_decay=1e-6, amsgrad=False)
optimizer_adain =   optim.AdamW(Fas_Net.parameters(), lr=1e-4, betas=(0.5, 0.8))#, eps=1e-08, weight_decay=1e-6, amsgrad=False)

Fas_Net.train()


data1_real = get_data_loader(data_path=live_path1, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data2_real = get_data_loader(data_path=live_path2, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data3_real = get_data_loader(data_path=live_path3, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data1_fake = get_data_loader(data_path=print_path1, data_path2=replay_path1,
                             batch_size=batch_size, shuffle=True)
data2_fake = get_data_loader(data_path=print_path2, data_path2=replay_path2,
                             batch_size=batch_size, shuffle=True)
data3_fake = get_data_loader(data_path=print_path3, data_path2=replay_path3,
                             batch_size=batch_size, shuffle=True)

iternum = max(len(data1_real), len(data2_real),
              len(data3_real), len(data1_fake),
              len(data2_fake), len(data3_fake))


print('iternum={}'.format(iternum))
data1_real = get_inf_iterator(data1_real)
data2_real = get_inf_iterator(data2_real)
data3_real = get_inf_iterator(data3_real)
data1_fake = get_inf_iterator(data1_fake)
data2_fake = get_inf_iterator(data2_fake)
data3_fake = get_inf_iterator(data3_fake)


T_transform = torch.nn.Sequential(
        T.Pad(40, padding_mode="symmetric"),
        T.RandomRotation(30),
        T.RandomHorizontalFlip(p=0.5),
        T.CenterCrop(256),
)

for epoch in range(5):

    for step in range(iternum):
        # ============ one batch extraction ============#
        img1_real, ls_lab1_real, m_lab1_real = next(data1_real)
        img1_fake, ls_lab1_fake, m_lab1_fake = next(data1_fake)
        
        d_lab1 = torch.ones_like(m_lab1_real).cuda()
        
        img2_real, ls_lab2_real, m_lab2_real = next(data2_real)
        img2_fake, ls_lab2_fake, m_lab2_fake = next(data2_fake)
        d_lab2 = torch.ones_like(m_lab1_real).cuda() * 2

        img3_real, ls_lab3_real, m_lab3_real = next(data3_real)
        img3_fake, ls_lab3_fake, m_lab3_fake = next(data3_fake)
        d_lab3 = torch.ones_like(m_lab1_real).cuda() * 3

        # ============ one batch collection ============# 
        catimg = torch.cat([img1_real, img2_real, img3_real,
                            img1_fake, img2_fake, img3_fake], 0).to(device_id)
        ls_lab = torch.cat([ls_lab1_real, ls_lab2_real, ls_lab3_real,
                            ls_lab1_fake, ls_lab2_fake, ls_lab3_fake], 0).to(device_id)  # .float()
        d_lab = torch.cat([d_lab1, d_lab2, d_lab3,
                           d_lab1, d_lab2, d_lab3], 0).to(device_id)

        batchidx = list(range(len(catimg)))
        random.shuffle(batchidx)

        img_rand = catimg[batchidx, :]
        ls_lab_rand = ls_lab[batchidx]
        d_lab_rand = d_lab[batchidx] - 1  # 1,2,3 ==> 0,1,2

        img_rand = T_transform(img_rand)
####################################################################
        # Learn_Original
        catfeat, p_liveness, p_domain, re_catfeat, \
        p_liveness_fwt, \
        p_liveness_hard = \
            Fas_Net(img_rand, update_step="Learn_Original")

        # original loss
        Loss_ls = criterion_ce(p_liveness.squeeze(), ls_lab_rand)
        
        # ladain sample loss
        Loss_ls_hard = criterion_ce(p_liveness_hard.squeeze(), ls_lab_rand)
        
        # disentangled loss
        Loss_cls_dm = criterion_ce(p_domain.squeeze(), d_lab_rand)
        
        # reconstruction loss
        Loss_re = criterionMSE(catfeat, re_catfeat)
        
        # fwt sample loss
        Loss_ls_fwt = criterion_ce(p_liveness_fwt.squeeze(), ls_lab_rand)

        Loss_all = Loss_ls + Loss_cls_dm + Loss_re + \
                   Loss_ls_hard + \
                   Loss_ls_fwt
                   
        optimizer_fas.zero_grad()
        Loss_all.backward()
        optimizer_fas.step()

        if (step + 1) % log_step == 0:
            print('[epoch %d step %d] Fixed_ FWT Loss_ls %.4f   Loss_cls_dm %.4f Loss_re %.4f '
                  ' Loss_ls_fwt %.4f  Loss_ls_hard %.4f'
                  % (epoch, step, Loss_ls.item(), Loss_cls_dm.item(),
                     Loss_re.item(), Loss_ls_fwt.item(),
                     Loss_ls_hard.item()))
####################################################################
        # Learn_AFT
        p_liveness_fwt, \
        f_liveness, f_liveness_fwt \
            = Fas_Net(img_rand, update_step="Learn_FWT")

        # fwt sample loss
        Loss_ls_fwt = criterion_ce(p_liveness_fwt.squeeze(), ls_lab_rand)
        
        # fwt sample and live sample similarity (encourage the dissimilarity)
        Loss_dissimilar = 1+torch.mean(criterion_cosine(f_liveness, f_liveness_fwt))
            
        Loss_all_fwt = Loss_ls_fwt + Loss_dissimilar
        optimizer_fwt.zero_grad()
        Loss_all_fwt.backward()
        optimizer_fwt.step()
        if (step + 1) % log_step == 0:
            print('[epoch %d step %d] Update FWT ,Loss_ls_fwt %.4f ,Loss_ls_dis %.4f'
                  % (epoch, step,  Loss_ls_fwt.item(), Loss_dissimilar.item()))
####################################################################
        # Learn_Adain
        p_liveness_hard \
            = Fas_Net(img_rand, update_step="Learn_Adain")
  
        # ladain sample loss with (1 - ls_lab_rand) (encourage strong domain augmentation)
        Loss_ls_hard = criterion_ce(p_liveness_hard.squeeze(), 1 - ls_lab_rand)   
        
        Loss_all_adain = Loss_ls_hard
        optimizer_adain.zero_grad()
        Loss_all_adain.backward()
        optimizer_adain.step()
        if (step + 1) % log_step == 0:
            print('[epoch %d step %d] Update LnAdaIN ,Loss_ls_hard %.4f'
                  % (epoch, step, Loss_ls_hard.item()))
####################################################################

        if ((step + 1) % model_save_step == 0):
            mkdir(results_path)
            save_index += 1
            torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                          "FASNet-{}.tar".format(save_index)))

    if ((epoch + 1) % model_save_epoch == 0):
        mkdir(results_path)
        save_index += 1
        torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                      "FASNet-{}.tar".format(save_index)))