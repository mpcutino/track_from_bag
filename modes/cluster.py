from sklearn.cluster import DBSCAN
import numpy as np


def dbscan_on_event_list(ev_list, eps=7, min_samples=10):
    return dbscan_on_list([(e.y, e.x) for e in ev_list], eps=eps, min_samples=min_samples)


def dbscan_on_list(param, eps=7, min_samples=10):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(param))


def get_clusters(events_list, pos_of_interest, eps=7, min_samples=10, use_unique_events=True):
    # use only the events from the interest class to build the clusters
    events_of_interest = [e for e in events_list if e.probs[pos_of_interest] == max(e.probs)]
    if len(events_of_interest):
        if use_unique_events:
            param = [(e.y, e.x) for e in events_of_interest]
            # get unique rows (not values). We need the index of the row, not the values
            _, indexes = np.unique(np.array(param), axis=0, return_index=True)
            events_of_interest = [events_of_interest[i] for i in indexes]
        clusters = dbscan_on_event_list(events_of_interest, eps=eps, min_samples=min_samples)
        # return build_img_from_clusters(events_of_interest, clusters, events_list)
        return clusters, events_of_interest
    return None, []


def get_rois_centers(cluster, ev_of_interest, all_events, pos_of_interest,
                            prob_filter=0, min_dims=None):
    """
    This draw the bounding boxes, but also filter clusters by probability and minimum dimensions
    :param cluster:
    :param ev_of_interest:
    :param all_events:
    :param pos_of_interest:
    :param prob_filter:
    :param min_dims: (height, width) corresponding with the minimum dimensions of the allowed bounding box
    :return:
    """
    # build region proposal from clusters
    rois = get_ROIs(cluster, [(e.x, e.y) for e in ev_of_interest], min_dims)
    # AFTER this, all list must have the same size as ROIS
    # and each position correspond to the rois coords in rois list

    # get the mean and sum probabilities for each class for every region
    rois_probs_sum, rois_probs_count = get_prob_inside_rois(rois, all_events)

    # classify the region using some strategy based on mean and sum values
    strategy = lambda x, y: x       # this strategy selects the sum as the important metric
    strategy = lambda x, y: np.array(x, dtype=np.float)/y    # this strategy selects the mean as the important metric
    classes, probs = classify_roi(rois_probs_sum, rois_probs_count, strategy)

    # filter rois
    filtered_rois_center = []
    for r, c, p in zip(rois, classes, probs):
        if p > prob_filter and (pos_of_interest is None or pos_of_interest == c):
            x, y, w, h = r
            filtered_rois_center.append((x + w/2, y + h/2))
    return filtered_rois_center


def classify_roi(prob_sum, prob_count, strategy):
    classes = []
    probs = []
    for s, c in zip(prob_sum, prob_count):
        value = strategy(s, c)
        category = np.argmax(value)
        prob = s[category] / c

        classes.append(category)
        probs.append(prob)
    return classes, probs


def get_ROIs(clusters, coord_of_interest, min_dims):
    rois = []
    if clusters is not None:
        all_xy_roi = {}
        for i, label in enumerate(clusters.labels_):
            if label == -1: continue
            x, y = coord_of_interest[i]
            if label not in all_xy_roi:
                all_xy_roi[label] = [], []
            all_xy_roi[label][0].append(x)
            all_xy_roi[label][1].append(y)
        for label in all_xy_roi:
            xs, ys = all_xy_roi[label]
            min_x, min_y = min(xs), min(ys)
            max_x, max_y = max(xs), max(ys)
            w = max_x - min_x
            h = max_y - min_y

            if min_dims and len(min_dims) == 2:
                if w < min_dims[0] or h < min_dims[1]:
                    # too small region
                    continue

            # save minimum x, minimum y, the width and height
            rois.append((min_x, min_y, w, h))
    return rois


def is_in_box(e, (x, y, w, h)):
    return (x <= e.x <= x + w) and (y <= e.y <= y + h)


def get_prob_inside_rois(rois, all_events):
    if len(all_events) and len(rois):
        rois_probs_sum = []
        rois_probs_count = []
        for _ in rois:
            rois_probs_sum.append(np.zeros_like(np.array(all_events[0].probs), dtype=np.float))
            rois_probs_count.append(0)
        for i, box in enumerate(rois):
            for e in all_events:
                if is_in_box(e, box):
                    rois_probs_sum[i] += np.array(e.probs)/100.0
                    rois_probs_count[i] += 1
        return rois_probs_sum, rois_probs_count
    return [], []
