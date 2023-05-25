import numpy as np
import torch


def evaluate_retrieval(image_feats, text_feats):
    """

    Args:
        image_feats: shape (N_datapoints, dim)
        text_feats: shape (N_datapoints, dim)

    Returns:

    """
    report = {}
    if isinstance(image_feats, torch.Tensor):
        image_feats = image_feats.detach().cpu().numpy()
    if isinstance(text_feats, torch.Tensor):
        text_feats = text_feats.detach().cpu().numpy()

    # image -> text
    sim_i2t = image_feats @ text_feats.T
    report_i2t = evaluate_similarity(sim_i2t)
    for k, v in report_i2t.items():
        report[f"i_{k}"] = v

    # text -> image
    sim_t2i = text_feats @ image_feats.T
    report_t2i = evaluate_similarity(sim_t2i)
    for k, v in report_t2i.items():
        report[f"t_{k}"] = v
    return report


def evaluate_similarity(similarity_matrix):
    n_datapoints = similarity_matrix.shape[0]
    ranks = np.empty(n_datapoints)

    # loop source embedding indices
    for index in range(n_datapoints):
        # get order of similarities to target embeddings
        inds = np.argsort(similarity_matrix[index])[::-1]

        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank

    # compute retrieval metrics
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    report_dict = {"r1": r1, "r5": r5, "r10": r10, "medr": medr, "meanr": meanr}
    return report_dict
