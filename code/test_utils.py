import torch
import numpy as np


def test_map(query_feature,query_label,gallery_feature, gallery_label):
    query_feature = query_feature / (query_feature.norm(dim=1, keepdim=True) + 1e-12) # normalize each row
    gallery_feature = gallery_feature / (gallery_feature.norm(dim=1, keepdim=True) + 1e-12) # normalize each row
    CMC = torch.IntTensor(len(gallery_label)).zero_() # https://pytorch.org/docs/stable/generated/torch.Tensor.zero_.html
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i],  gallery_feature, gallery_label) # the first two arguments are a caption and its correct ID

        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0], CMC[4], CMC[9], ap / len(query_label)

def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1) # reshape to 1 column (2048x1)
    score = torch.mm(gf, query) # gf is (max_sizex2048), score is (max_size,1)
    score = score.squeeze(1).cpu() # (max_size,)
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1] # score argsorted in decreasing order
    # gl=gl.cuda().data.cpu().numpy()
    gl=gl.cpu().numpy()
    # ql=ql.cuda().data.cpu().numpy()
    ql=ql.cpu().numpy()
    query_index = np.argwhere(gl == ql) # gl (max_size,2048) ql (1x2048), ql is broadcasted to every row
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp


def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc