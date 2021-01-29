from os import listdir, path as os_path

import cv2
import rosbag
import genpy

from utils.constants import PROB_EVENT_TOPIC, IMG_H, IMG_W
from utils.img_utils import get_cluster_bbox_center
from modes.cluster import get_clusters, get_rois_centers


class Modified_MainWindow:

    def __init__(self, bag_file, images_folder):
        self.bag_messages = []
        self.bag_msg_count = 0
        self.images = []
        self.images_folder = images_folder

        self.load_file(bag_file)

        self.color_map = "viridis"
        # ["cluster", "prob_pixel", "binary"]
        self.mode = "cluster"
        # ["pixels", "events"]
        self.group_by = "events"
        self.prob_filter = 50
        self.minh = 5
        self.minw = 5
        self.eps = 15
        self.minPoints = 50
        self.use_cluster_prob = True

    def load_file(self, filename):
        if len(str(filename)):
            bag = rosbag.Bag(str(filename))
            bag_messages = list(bag.read_messages(PROB_EVENT_TOPIC))
            if len(bag_messages):
                print(len(bag_messages))
                self.bag_messages = bag_messages
                self.images = []
            else:
                print("Not a valid bag. It must contains the topic {0}".format(PROB_EVENT_TOPIC))

    def make_video(self):
        if len(self.bag_messages):

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output.avi', fourcc, 40.0, (IMG_W, IMG_H))

            for img_name in sorted(listdir(self.images_folder)):
                if not img_name.endswith(".png"): continue

                img = cv2.imread(os_path.join(self.images_folder, img_name))
                f = img_name.split(".")[0]
                sec, n_sec = int(f[:10]), float(f[10:])
                ts = genpy.Time(secs=sec, nsecs=n_sec)
                events = self.get_new_latest_events(ts)
                center = self.get_cluster_center(events)

                # some images does not have this resolution
                img = cv2.resize(img, (IMG_W, IMG_H))
                if len(center):
                    for c in center:
                        cv2.circle(img, c, 2, (0, 255, 0), -1)
                out.write(img)
                print("Processed image {0}".format(img_name))

            cv2.destroyAllWindows()
            out.release()

    def get_cluster_center(self, events):
        eps = self.eps
        min_samples = self.minPoints
        group_by_pixels = self.group_by == "pixels"
        pos_of_interest = 0
        clusters, ev_of_interest = get_clusters(events, pos_of_interest, eps=eps,
                                                min_samples=min_samples, use_unique_events=group_by_pixels)
        # return get_cluster_bbox_center(clusters, ev_of_interest)
        return get_rois_centers(clusters, ev_of_interest, events, pos_of_interest)

    def get_new_latest_events(self, end_TS):
        latest_events = []
        for msg in self.bag_messages[self.bag_msg_count:]:
            if msg.timestamp > end_TS:
                break
            self.bag_msg_count += 1
            for e in msg.message.events:
                latest_events.append(e)
        return latest_events

    @staticmethod
    def get_binary_prob_value(e):
        # persons are index 0 in prob array
        if e.probs[0] == max(e.probs):
            return 1
        return 0

    @staticmethod
    def get_colored_prob_value(e):
        # persons are index 0 in prob array
        return e.probs[0]/100.0


if __name__ == "__main__":
    img_folder = "/media/mpcutino/ed230f55-7d2a-4a71-87fc-84b1b958f0f1/migue/Proyectos/build_dataset_fromYOLO/" \
                 "data_bBox/person/Record01/images/clean_images"
    bag_f = "/media/mpcutino/ed230f55-7d2a-4a71-87fc-84b1b958f0f1/migue/Proyectos/nn_events_YOLO/" \
            "Record01_classified.bag"
    #bag_f = "/media/mpcutino/ed230f55-7d2a-4a71-87fc-84b1b958f0f1/migue/Proyectos/build track video/test.bag"
    ui = Modified_MainWindow(bag_f, images_folder=img_folder)
    ui.make_video()

