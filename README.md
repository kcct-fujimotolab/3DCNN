#3DCNN
This is an Inplementation of 3D Convolutional Neural Network for video classification using [Keras](https://keras.io/)(with [tensorflow](https://www.tensorflow.org/) as backend).

##Description
This code uses [UCF-101 dataset](http://crcv.ucf.edu/data/UCF101.php).
This code generates graphs of accuracy and loss, plot of model, result and class names as txt file and model as hd5 and json.

##Options
`--batch`   batch size, default is 128  
`--epoch`   the number of epochs, default is 100  
`--videos`  the name of directory where dataset is stored, default is UCF101  
`--nclass`  the number of classes you want to use, default is 101  
`--output`  directory where the results described above will be saved  
`--color`   use RGB image or grayscale image, default is False  
`--skip`    get frame at interval or contenuously, default is True  
`--depth`   the number of frames to use, default is 10

##Demo
You can execute like the following.
```sh
python 3dcnn.py --batch 32 --epoch 50 --videos dataset/ --nclass 10 --output 3dcnnresult/ --color True --skip False --depth 15
```

##Other files
`2dcnn.py`  2DCNN model  
`display.py` get example images from the dataset.  
`videoto3d.py`  get frames from a video, extract a class name from filename of a video in UCF101.  
