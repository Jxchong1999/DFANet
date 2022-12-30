
from models.DFANet import *
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import torch


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def NormalizeData(data):
    if np.min(data) ==np.max(data):
        a = data
    else:
        a = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    return  a


test_dataset = "Oulu"
live_path = '/shared/domain-generalization/'+test_dataset+'_images_live.npy'
spoof_path = '/shared/domain-generalization/'+test_dataset+'_images_spoof.npy'
live_data = np.load(live_path)
spoof_data = np.load(spoof_path)
live_label = np.ones(len(live_data), dtype=np.int64)
spoof_label = np.zeros(len(spoof_data), dtype=np.int64)

total_data = np.concatenate((live_data, spoof_data), axis=0)
total_label = np.concatenate((live_label, spoof_label), axis=0)

print(live_data.shape, len(total_data))

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))
# dataloader
data_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=40,
                                          shuffle=False, )
device_id = "cuda:0" 
FASNet = Ad_LDCNet().to(device_id)

model_path = "/home/Jxchong/adversarial/resultpath/fwt(both)(o)/FASNet-"

print("model_path", model_path)
print("live_path", live_path)
print("spoof_path", spoof_path)
 
for epoch in range(1,80):
    FASNet_path = model_path + str(epoch) + ".tar"


    FASNet.load_state_dict(torch.load(FASNet_path, map_location=device_id),strict=False) 
    FASNet.eval()

    score_list_ori = []
    score_list_spoof = []
    Total_score_list_cs = []
    label_list = []
    TP = 0.0000001
    TN = 0.0000001
    FP = 0.0000001
    FN = 0.0000001

    for i, data in enumerate(data_loader, 0):
        images, labels = data
        images = images.to(device_id)
        label_pred = FASNet(images)

        score = F.softmax(label_pred, dim=1).cpu().data.numpy()[:, 1]  # multi class

        for j in range(images.size(0)):
            score_list_ori.append(score[j]) 
            label_list.append(labels[j])



    score_list_ori = NormalizeData(score_list_ori)

    score_list_spoof = [1 - score_list_spoof[i] for i in range(len(score_list_spoof))]
    for i in range(0, len(label_list)):
        Total_score_list_cs.append(score_list_ori[i] * 0.1) 
        if score_list_ori[i] * 0.1 == None:
            print(score_list_ori[i] * 0.1)
    fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, Total_score_list_cs)
    threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

    for i in range(len(Total_score_list_cs)):
        score = Total_score_list_cs[i]
        if (score >= threshold_cs and label_list[i] == 1):
            TP += 1
        elif (score < threshold_cs and label_list[i] == 0):
            TN += 1
        elif (score >= threshold_cs and label_list[i] == 0):
            FP += 1
        elif (score < threshold_cs and label_list[i] == 1):
            FN += 1

    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP)
    


    print("Epoch", epoch, "CL",
          "ACER", np.round((APCER + NPCER) / 2, 4),
          "AUC", roc_auc_score(label_list, Total_score_list_cs))

