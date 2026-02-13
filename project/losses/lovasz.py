import torch
import torch.nn.functional as F

def flatten_binary_scores(scores, labels):
    scores = scores.view(-1)
    labels = labels.view(-1)
    return scores, labels

def lovasz_hinge(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    labels = (labels > 0.5).float()
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    labels_sorted = labels[perm]
    grad = lovasz_grad(labels_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_grad(gt_sorted):
    p = gt_sorted.sum()
    intersect = p - gt_sorted.cumsum(0)
    union = p + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersect / union
    if gt_sorted.numel() > 1:
        grad = jaccard[1:] - jaccard[:-1]
        grad = torch.cat([jaccard[:1], grad], dim=0)
    else:
        grad = jaccard
    return grad
