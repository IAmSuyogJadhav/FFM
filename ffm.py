import torch
from pytorch_grad_cam import GradCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import numpy as np

# batched dice score
def batched_dice(pred, target, eps=1e-5):
    intersection = (pred * target).sum(axis=(1, 2))
    union = pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def compute_ffm(model, img, gt_label, target_layer, reduce='mean', top_k=3, threshold=0.5, eps=1e-5, device='cuda'):
    """
    Compute Feature Faithfulness Metric (FFM) for a single image.

    Args:
        model: PyTorch model
        img: Image tensor [B, C, H, W]
        gt_label: Ground truth label for each of the images in the batch [B]
        target_layer: Target layer for computing CAMs. Has to be a layer in the model.
        reduce: Reduction strategy for computing FFM. Default: 'mean'. Options: 'mean', 'none', None.
        top_k: Number of top-k labels to consider for computing. Default: 3
        threshold: Threshold for binarizing the CAMs. Default: 0.5
        eps: Epsilon value for numerical stability. Default: 1e-5
        device: Device to run the model on. Default: 'cuda'

    Returns:
        ffm: Feature Faithfulness Metric (FFM) score for the images in the batch.
            Noemlised to [0, 1] where 1 means perfect faithfulness.

    """
    # Sanity checks
    B = img.shape[0]
    assert B == len(gt_label), "Number of images and ground truth labels should be same. Make sure you are passing a batch of images."

    # Get model confidence scores for all labels
    model.eval()
    model.to(device)
    img = img.to(device)
    with torch.no_grad():
        output = model(img).softmax(dim=1)
        all_scores, all_labels = torch.topk(output, len(output[0]))  # [B, num_classes]
        all_scores = all_scores.detach().cpu().numpy(); #print('all_scores', all_scores.shape) # [B, num_classes]
        all_labels = all_labels.detach().cpu().numpy(); #print('all_labels', all_labels.shape)  # [B, num_classes]

    # Store gt scores in a separate array and zero it in all_scores so that it doesn't count
    gt_scores = [] # [B]
    for i in range(len(gt_label)):
        label_idx = all_labels[i].tolist().index(gt_label[i])
        gt_scores.append(all_scores[i][label_idx])
        all_scores[i][label_idx] = 0

    # Get top-k labels and scores, and ground truth scores
    top_k_labels = all_labels[:, :top_k]; #print('top_k_labels', top_k_labels.shape)
    top_k_scores = all_scores[:, :top_k]; #print('top_k_scores', top_k_scores.shape)

    # Get CAMs for top-k labels and gt labels
    # print(top_k, B)
    input_tensor = torch.vstack([img]*(top_k + 1)).to(device); #print('input_tensor', input_tensor.shape)  #[top_k*B + B, C, H, W]
    targets = [ClassifierOutputTarget(l) for l in top_k_labels.flatten()] + [ClassifierOutputTarget(l) for l in gt_label]; #print('targets', len(targets))  #[top_k*B + B]
    G = GradCAM(model=model, target_layers=[target_layer])
    A = AblationCAM(model=model, target_layers=[target_layer])
    gcam = G(input_tensor=input_tensor, targets=targets); #print('gcam', gcam.shape)
    acam = A(input_tensor=input_tensor, targets=targets); #print('acam', acam.shape)

    # Separate CAMs for top-k labels and gt labels
    gcam_top_k = gcam[:top_k*B]; #print('gcam_top_k', gcam_top_k.shape)
    gcam_top_k = gcam_top_k.reshape(B, top_k, *gcam_top_k.shape[1:]); #print('gcam_top_k', gcam_top_k.shape)
    acam_top_k = acam[:top_k*B]; #print('acam_top_k', acam_top_k.shape)
    acam_top_k = acam_top_k.reshape(B, top_k, *acam_top_k.shape[1:]); #print('acam_top_k', acam_top_k.shape)

    gcam_gt = gcam[top_k*B:]
    acam_gt = acam[top_k*B:]; #print('acam_gt', acam_gt.shape)

    # Threshold CAMs
    gcam_top_k = (gcam_top_k > threshold)  # [top_k*B, H, W]
    gcam_gt = (gcam_gt > threshold)  # [B, H, W]
    acam_top_k = (acam_top_k > threshold)  # [top_k*B, H, W]
    acam_gt = (acam_gt > threshold)  # [B, H, W]

    # FFM
    ffm_gt = gt_scores * batched_dice(gcam_gt, acam_gt, eps=eps); #print('ffm_gt', ffm_gt)
    ffm_others = - np.sum(top_k_scores.sum(axis=1) * np.array([batched_dice(gcam_top_k[i], acam_top_k[i], eps=eps).mean() for i in range(B)]))

    # print(gt_scores, top_k_scores.sum(axis=1).shape, top_k_scores.shape)
    norm_factor = gt_scores + top_k_scores.sum(axis=1); #print('norm_factor', norm_factor)
    ffm = (ffm_gt + ffm_others) / norm_factor

    # Reduction
    if reduce == 'mean':
        ffm = ffm.mean()
    elif reduce == 'none' or reduce is None:
        pass
    else:
        raise ValueError(f"Invalid value for reduce: {reduce}. Expected: 'mean', 'none', None.")

    # Normalize to 0-1
    ffm = (ffm + 1) / 2

    return ffm