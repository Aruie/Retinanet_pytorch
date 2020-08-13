import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module) :
    def __init__(self, gamma = 2, alpha = 0.25) :
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, classifications, regressions, anchors, annotations) :
        
        
        batch = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0,:,:]

        anchor_widths = anchor[:,2] - anchor[:,0]
        anchor_heigths = anchor[:,3] - anchor[:,1]
        anchor_ctr_x = anchor[:,0] + 0.5 * anchor_widths
        anchor_ctr_x = anchor[:,1] + 0.5 * anchor_heigths
        
        for j in range(batch):

            classification = classifications[j, :, :] 
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            # 데이터셋에 -1 -1 -1 -1 -1 로 나온데이터가 꽤많은데 이걸 삭제
            # 이유를 모르겠.... 다른 태스크용인가
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            # 할당된 타겟이 없을경우 ( 백그라운드로 돌림 )
            if bbox_annotation.shape[0] == 0:
                # - α(1-pt)^γ * log(pt)
                alpha_factor = torch.ones(classification.shape).to(self.device) * self.alpha

                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

                bce = -(torch.log(1.0 - classification))

                cls_loss = focal_weight * bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float()).to(self.device)
                continue
            
            # 모든 앵커와 GT Box의 IoU 계산 (anchor * class)
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) 

            # 클래스별 가장 IoU가 큰 앵커 구함 (class) * 1)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) 

            # 전부 -1로만듬 ( -1은 사용안함의 의미 )
            targets = torch.ones(classification.shape) * -1
            targets = targets.to(self.device)

            # 0.4보다 작은거 0으로
            targets[torch.lt(IoU_max, 0.4), :] = 0

            # 0.5보다 크면서 가장큰 것들 구하고 갯수구함
            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]
            print(assigned_annotations[positive_indices, 4].long().shape)


            targets[positive_indices, :] = 0
            #targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            # - α(1-pt)^γ * log(pt)
            alpha_factor = torch.ones(targets.shape).to(self.device) * self.alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(self.device))
            
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(self.device)

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float()).to(self.device)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)



def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU