import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.epsilon = args.epsilon
        self.margin = getattr(args, 'margin', 0.5)  # default value is 0.5

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon)) # (4)
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        return cmpm_loss

    def compute_triplet_loss(self, image_embeddings, text_embeddings, labels):
        batch_size = image_embeddings.shape[0]
        
        # Compute pairwise distances between image and text embeddings
        dist_matrix = torch.norm(image_embeddings[:, None] - text_embeddings, dim=2)
        
        labels_eq = (labels[:, None] == labels).float() - torch.eye(batch_size, device=labels.device)
        positive_dist = (dist_matrix * labels_eq).max(dim=1)[0]
        
        labels_ne = (labels[:, None] != labels).float()
        negative_dist = (dist_matrix + labels_eq * 1e5).min(dim=1)[0]

        triplet_loss = (positive_dist - negative_dist + self.margin).clamp(min=0).mean()
        return triplet_loss

    def forward(self, img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
                txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels):
        loss_cmpm = 0.0
        loss_triplet = 0.0

        for img, txt in zip([img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f4],
                            [txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f4]):
            loss_cmpm += self.compute_cmpm_loss(img, txt, labels)
            loss_triplet += self.compute_triplet_loss(img, txt, labels)

        # You can weigh the importance of the two losses as needed. Here, I assume they are equally important.
        loss = loss_cmpm + loss_triplet

        return loss
