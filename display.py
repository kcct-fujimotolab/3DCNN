import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from videoto3d import Videoto3D


def main():
    vid3d = Videoto3D(256, 256, 1)
    flist = os.listdir('UCF101')
    #flist.remove('.DS_Store')
    labels = list(set([vid3d.get_UCF_classname(x) for x in flist]))

    plt.figure(figsize=(7, 10))
    cnt = 0
    while cnt < 15:
        print(cnt)
        if flist[0].find(labels[cnt]) >= 0:
            print(flist[0])
            plt.subplot(5, 3, cnt + 1)
            plt.axis('off')
            img = vid3d.video3d(os.path.join('UCF101', flist[0]))
            plt.imshow(img[0], cmap='gray')
            print(img.shape)
            plt.title(vid3d.get_UCF_classname(flist[0]))
            cnt += 1
        del flist[0]
    plt.savefig('display.png')

if __name__ == '__main__':
    main()
