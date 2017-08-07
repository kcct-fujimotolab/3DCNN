import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import videoto3d
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def loaddata(video_dir, vid3d, nclass, result_dir):
    files = os.listdir(video_dir)
    X = []
    labels = []
    labellist = []

    pbar = tqdm(total=len(files))

    for filename in files:
        if filename == '.DS_Store':
            continue
        name = os.path.join(video_dir, filename)
        #print('loading video:{}'.format(name))
        label = vid3d.get_UCF_classname(filename)
        # print('label:{}'.format(label))
        if label not in labellist:
            if len(labellist) >= nclass:
                continue
            labellist.append(label)
        labels.append(label)
        X.append(vid3d.video3d(name))
        pbar.update(1)

    pbar.close()
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    return np.array(X), labels


def main():
    parser = argparse.ArgumentParser(description='2D convolution')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='videos',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=101)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    img_rows, img_cols = 32, 32

    vid3d = videoto3d.Videoto3D(img_rows, img_cols, 1)
    x, y = loaddata(args.videos, vid3d, args.nclass, args.output)
    X = x.reshape(x.shape[0], img_cols, img_rows, 1)
    nb_classes = max(y) + 1
    Y = np_utils.to_categorical(y, nb_classes)
    X = X.astype('float32')
    print('X shape:{}\nYshape:{}'.format(X.shape, Y.shape))

    # define model
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, to_file=os.path.join(args.output, 'model.png'))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=4)

    history = model.fit(X_train, Y_train,
                        batch_size=args.batch,
                        nb_epoch=args.epoch,
                        validation_data=(X_test, Y_test),
                        shuffle=True)
    model_json = model.to_json()
    with open(os.path.join(args.output, 'ucf101cnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'ucf101cnnmodel.hd5'))

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(history, args.output)
    save_history(history, args.output)

if __name__ == '__main__':
    main()
