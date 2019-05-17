import numpy as np, pandas as pd
import cv2
from tqdm import tqdm_notebook, tqdm # Iteration visualization
tqdm.pandas(desc="Loading") # to do progress_apply for pandas


def extr_data_txt(path):
    """
    Load data from text file.
    """
    with open(path, "r") as f:
        data = []
        for itr, line in tqdm_notebook(enumerate(f)):
            # Because we got annotation in the first two lines
            if itr >= 2:
                data.append(line.split())
    return data


train_test_valid_anot = pd.DataFrame(extr_data_txt('list_eval_partition.txt'),
                                     columns=['Path', 'type'])
train_test_valid_anot.to_csv('train_test_valid_anot.csv', index=False)
categories_img = pd.DataFrame(extr_data_txt('list_category_img.txt'),
                              columns=['Path', 'cat'])
categories_img['cat'] = categories_img['cat'].apply(lambda x: int(x)-1)  # Categories starts with 1, fixing it.
categories_img.to_csv('categories_img.csv', index=False)
box_img = pd.DataFrame(extr_data_txt('list_bbox.txt'),
                       columns=['Path', 'x1', 'y1', 'x2', 'y2'])
box_img.to_csv('bbox_img.csv', index=False)


def get_img_shape(path):
    path = path
    img = cv2.imread(path)
    try:
        return img.shape
    except AttributeError:
        print('error! ', path)
        return None, None, None


def convert_labels(path, x1, y1, x2, y2):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
    """
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    size = get_img_shape(path)
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h

box_img['x'], box_img['y'], box_img['width'], box_img['height'] = zip(*box_img.progress_apply(lambda row: convert_labels(row['Path'], row['x1'], row['y1'], row['x2'], row['y2']), axis=1))
# Like python for one lone code.
box_img.to_csv('box_img.csv', index=False)


box_img = pd.read_csv('box_img.csv')
df = box_img.merge(train_test_valid_anot).merge(categories_img)
df.to_csv('annotation_w-o_atr.csv', index=False)


np.savetxt('train.txt', df[df['type'] == 'train'].values, fmt='%s')
np.savetxt('helloflask.txt', df[df['type'] == 'helloflask'].values, fmt='%s')
np.savetxt('val.txt', df[df['type'] == 'val'].values, fmt='%s')


def save_txt(name, text):
    with open(name, 'w+') as f:
        f.write(text)


def make_img_txt(t='train'):
    path_change = []
    for itr, path in tqdm_notebook(enumerate(df[df['type'] == t]['Path'].values)):
        image = cv2.imread(path)
        name = t + '/' + str(path[4:])
        path_change.append(path)
        cv2.imwrite(name, image)

        arr = df[df.Path == path][['cat', 'x', 'y', 'width', 'height']].astype(str).values.flatten()
        save_txt(name[0:-4] + '.txt', ' '.join(arr))

        np.savetxt(t + '.txt', path_change, fmt='%s')


# make_img_txt('train')
make_img_txt('val')