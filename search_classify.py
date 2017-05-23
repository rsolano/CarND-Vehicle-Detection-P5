import cv2
import numpy as np
import time

from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from lesson_functions import *


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    detections = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                detections.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return detections, draw_img

def process_data():
    # Read in cars and notcars
    cars = glob.glob('data/vehicles/**/*.png', recursive=True)
    notcars = glob.glob('data/non-vehicles/**/*.png', recursive=True)

    car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    return X_train, X_test, y_train, y_test, X_scaler

def train(X_train, X_test, y_train, y_test):
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check some predictions
    print('labels    :', y_test[0:10])
    print('prediction:', svc.predict(X_test[0:10]))
    return svc

def pipeline(image):
    box_list, bbox_img = find_cars(image, ystart, ystop, scale, svc, scaler, orient,
                        pix_per_cell, cell_per_block, spatial_size, hist_bins)

    if DEBUG:
        print(box_list)
    saved_detections.append(box_list)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    for detections in saved_detections:
        heat = add_heat(heat, detections)
    threshold = 1 + len(saved_detections)/2
    print("Threshold:", threshold)
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    if DEBUG:
        return draw_img, heatmap, labels[0], bbox_img

    return draw_img


if __name__ == '__main__':
    import glob
    import matplotlib.image as mpimg
    import pickle

    from collections import deque

    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    # Min and max in y to search
    ystart = 400
    ystop = 656
    scale = 1.5

    saved_detections = deque(maxlen=20)

    SAVED_MODEL = 'saved_model.p'
    DEBUG = False
    RENDER_CLIP = True

    try:
        with open(SAVED_MODEL, "rb") as f:
            print("Using pre-trained model...")
            model = pickle.load(f)
            svc = model["svc"]
            scaler = model["scaler"]
    except:
        print("Extracting training data and training model...")
        X_train, X_test, y_train, y_test, scaler = process_data()
        svc = train(X_train, X_test, y_train, y_test)

        model = {}
        model["svc"] = svc
        model["scaler"] = scaler
        with open(SAVED_MODEL, "wb") as f:
            pickle.dump(model, f)

    if RENDER_CLIP:
        from moviepy.editor import VideoFileClip
        _input = 'project_video.mp4'
        _output = 'project_video_out.mp4'
        # _input = 'test_video.mp4'
        # _output = 'test_video_out04.mp4'

        clip = VideoFileClip(_input).subclip(25, 30)
        out_clip = clip.fl_image(pipeline)
        out_clip.write_videofile(_output, audio=False)
    else:
        # Search the test images and draw the results
        test_files = glob.glob('test_images/test?.jpg')
        for file, i in zip(test_files, range(1, len(test_files) + 1)):
            print(file)
            image = mpimg.imread(file)
            if DEBUG:
                window_img, heatmap, labels, bboxes = pipeline(image)
                mpimg.imsave('output_images/test_out{}.png'.format(i), window_img)
                mpimg.imsave('output_images/test_heatmap_out{}.png'.format(i), heatmap, cmap='hot')
                mpimg.imsave('output_images/test_labels_out{}.png'.format(i), labels, cmap='gray')
                mpimg.imsave('output_images/test_bboxes_out{}.png'.format(i), bboxes)
            else:
                window_img = pipeline(image)
                mpimg.imsave('output_images/test_out{}.png'.format(i), window_img)
