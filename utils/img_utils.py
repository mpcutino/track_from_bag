import cv2
import numpy as np

from utils.constants import IMG_H, IMG_W


def draw_bBox(img, start, end, class_name, prob, bbox_color=(0, 255, 0)):
    """
    draw bounding box in image.
    start is a tuple of (x, y) top left pixel
    end is a tuple of bottom right pixel
    """
    v = cv2.rectangle(img, start, end, bbox_color, 1)

    text = "{0}: {1:.2f}%".format(class_name, prob) if class_name != "" else "{0:.1f}%".format(prob)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)[0]
    cv2.rectangle(img, (start[0], start[1]-text_size[1]), (start[0]+text_size[0], start[1]), bbox_color, cv2.FILLED)
    cv2.putText(img, text, start, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)


def draw_bBox_from_cluster(cluster, ev_of_interest, all_events, pos_of_interest,
                           prob_filter=0, min_dims=None, use_cluster_prob=False):
    """
    This draw the bounding boxes, but also filter clusters by probability and minimum dimensions
    :param cluster:
    :param ev_of_interest:
    :param all_events:
    :param pos_of_interest:
    :param prob_filter:
    :param use_cluster_prob
    :param min_dims: (height, width) corresponding with the minimum dimensions of the allowed bounding box
    :return:
    """
    cluster_img = build_img_from_clusters(ev_of_interest, cluster)
    full_img = build_prob_img(all_events, pos_of_interest)

    probs = np.unique(cluster_img)
    for p in probs:
        if p == 0: continue
        h_indexes, w_indexes = np.where(cluster_img == p)
        start = w_indexes.min(), h_indexes.min()
        end = w_indexes.max(), h_indexes.max()

        box_h = end[1] - start[1]
        box_w = end[0] - start[0]
        h, w = min_dims if min_dims and len(min_dims) == 2 else (box_h, box_w)

        if use_cluster_prob:
            true_prob = p
        else:
            # instead of using prob in cluster (that ony consider elements of the cluster),
            # use the full image in the box slice to compute the probability inside the box
            box_slice = full_img[start[1]:end[1], start[0]:end[0]]
            true_prob = box_slice.sum()/(box_slice > 0).sum()

        if true_prob >= prob_filter and h <= box_h and w <= box_w:
            draw_bBox(full_img, start, end, "", true_prob*100, bbox_color=128/255.0)
        # else:
        #     full_img[h_indexes, w_indexes] = 0
    return full_img


def build_img_from_clusters(events_of_interest, clusters):
    new_img = np.zeros((IMG_H, IMG_W))
    if clusters is not None:
        labels_count = clusters.labels_.max() + 1    # because is zero indexed
        label_prob, label_count = np.zeros(labels_count), np.zeros(labels_count)
        for i, label in enumerate(clusters.labels_):
            if label == -1: continue    # outliers
            event = events_of_interest[i]
            label_prob[label] += event.probs[0]
            label_count[label] += 1
        # get probabilities for each label with a value between 0 and 1
        prob_label = label_prob/(100.0*label_count)
        for i, label in enumerate(clusters.labels_):
            if label == -1: continue    # outliers
            event = events_of_interest[i]
            # label increased by one to difference between label 0 and number 0 in the matrix
            new_img[event.y, event.x] = prob_label[label]
    return new_img


def build_prob_img(events, index_of_interest):
    new_image = np.zeros((IMG_H, IMG_W))
    for ev in events:
        new_image[ev.y, ev.x] = ev.probs[index_of_interest]/100.0
    return new_image


def get_cluster_bbox_center(cluster, ev_of_interest):
    cluster_img = build_img_from_clusters(ev_of_interest, cluster)
    centers = []

    probs = np.unique(cluster_img)
    for p in probs:
        if p == 0: continue
        # la probabilidad de dos etiquetas puede ser la misma, asi que esto no es del todo correcto
        h_indexes, w_indexes = np.where(cluster_img == p)
        start = w_indexes.min(), h_indexes.min()
        end = w_indexes.max(), h_indexes.max()

        box_h = end[1] - start[1]
        box_w = end[0] - start[0]

        center = start[0] + box_w/2, start[1] + box_h/2
        centers.append(center)
    return centers
