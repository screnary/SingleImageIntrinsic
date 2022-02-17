import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib._png import read_png
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import os
import sys
import argparse
import pdb

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument('--clsname', default='v8_3')
parser.add_argument('--ver', default='v0')
FLAGS = parser.parse_args()

CLASSNAME = FLAGS.clsname
ver = 'log-'+FLAGS.ver


# Scale and visualize the embedding vectors
def plot_embedding(X, img_names, title=None, size=(80,80)):
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
    X = X
    plt.figure(figsize=size)
    ax = plt.subplot(111)
    """ grid coordinates """
    g_size = 50
    grid = 1.0/float(g_size)
    locations_ = (X / grid).astype(int)
    locations_ = locations_ / float(g_size)
    x = locations_[:,0]
    idx_x = np.argsort(x, axis=0)
    locations_ = locations_[idx_x]
    y = locations_[:,1]
    idx_y = np.argsort(-y, axis=0)
    locations = locations_[idx_y]
    idx = idx_x[idx_y]

    # locations = locations_
    # pdb.set_trace()

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(locations.shape[0]):
            dist = np.sum((locations[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-5:  # 4e-3
                # don't show points that are too close
                continue
            shape_img = get_shapeimg(img_names[idx[i]])
            shown_images = np.r_[shown_images, [locations[i]]]  # the coordinate of shown images
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(shape_img, zoom=0.23),  # 0.23
                locations[i], frameon=False)
            ax.add_artist(imagebox)
    # vis_x = X[:,0]
    # vis_y = X[:,1]
    # ax.scatter(vis_x, vis_y, s=5)
    # ax.scatter(locations[:,0], locations[:,1], s=20)
    plt.xticks([]), plt.yticks([])
    # # hide boundary
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if title is not None:
        plt.title(title)


# get images for visualize the 3D shapes
def get_shapeimg(img_name):
    return read_png(img_name)


def two_stream_vis():
    feature_fn_r = os.path.join('./features', CLASSNAME, ver, 'feature_r.npy')
    feature_fn_s = os.path.join('./features', CLASSNAME, ver, 'feature_s.npy')
    feature_r = np.load(feature_fn_r)  # (704,512)
    feature_s = np.load(feature_fn_s)  # (704,512)
    # feature = feature_r
    feature = np.concatenate([feature_r, feature_s], axis=0)

    tsne = TSNE(n_components=2, perplexity=18, init='random', learning_rate=35,
                early_exaggeration=13.0,
                n_iter=3000, random_state=501)  # perplexity 20, early_exaggeration 80
    # scipy.spatial.distance.pdist
    fea_tsne = tsne.fit_transform(feature)
    x_min, x_max = np.min(fea_tsne, 0), np.max(fea_tsne, 0)
    X = (fea_tsne - x_min) / (x_max - x_min) * 1
    vis_x_r = X[0:150, 0]
    vis_y_r = X[0:150, 1]
    # vis_z_r = X[100:200, 2]
    vis_x_s = X[-300:-150, 0]
    vis_y_s = X[-300:-150, 1]

    mean_x_r = np.mean(vis_x_r)
    mean_y_r = np.mean(vis_y_r)
    std_x_r = np.std(vis_x_r)
    std_y_r = np.std(vis_y_r)

    mean_x_s = np.mean(vis_x_s)
    mean_y_s = np.mean(vis_y_s)
    std_x_s = np.std(vis_x_s)
    std_y_s = np.std(vis_y_s)

    vis_x_r_new = (vis_x_r - mean_x_r) / std_x_r * std_x_s + mean_x_r
    vis_y_r_new = (vis_y_r - mean_y_r) / std_y_r * std_y_s + mean_y_r
    # vis_z_s = X[-200:-100, 2]

    plt.figure()
    # ax = plt.subplot(111, projection='3d')
    ax = plt.subplot(111)
    plt.scatter(vis_x_r_new, vis_y_r_new, s=15, c='b', marker='v')
    # plt.scatter(vis_x_r, vis_y_r, s=15, c='b', marker='v')
    plt.scatter(vis_x_s, vis_y_s, s=35, c='g', marker='+')
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    feature_fn_r = os.path.join('./features', CLASSNAME, ver, 'feature_r.npy')
    feature_fn_s = os.path.join('./features', CLASSNAME, ver, 'feature_s.npy')
    feature_r = np.load(feature_fn_r)  # (704,512)
    feature_s = np.load(feature_fn_s)  # (704,512)
    feature = feature_r
    # feature = feature_s

    tsne = TSNE(n_components=2, perplexity=18, init='pca', learning_rate=35,
                early_exaggeration=13.0,
                n_iter=3000, random_state=501)  # perplexity 20, early_exaggeration 80
    # scipy.spatial.distance.pdist
    fea_tsne = tsne.fit_transform(feature)
    x_min, x_max = np.min(fea_tsne, 0), np.max(fea_tsne, 0)
    X = (fea_tsne - x_min) / (x_max - x_min) * 1
    vis_x_r = X[0:300,0]
    vis_y_r = X[0:300,1]
    # vis_z_r = X[100:200, 2]

    # vis_z_s = X[-200:-100, 2]

    plt.figure()
    # ax = plt.subplot(111, projection='3d')
    ax = plt.subplot(111)
    # plt.scatter(vis_x_r_new, vis_y_r_new, s=15, c='b', marker='v')
    plt.scatter(vis_x_r, vis_y_r, s=15, c='c', marker='o')
    # plt.scatter(vis_x_s, vis_y_s, s=35, c='g', marker='+')
    plt.show()
    pdb.set_trace()
