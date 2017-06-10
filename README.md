# 3DCNN
 Inplementation of 3D Convolutional Neural Network for video classification using [Keras](https://keras.io/)(with [tensorflow](https://www.tensorflow.org/) as backend).

## Description
This code requires [UCF-101 dataset](http://crcv.ucf.edu/data/UCF101.php).
This code generates graphs of accuracy and loss, plot of model, result and class names as txt file and model as hd5 and json.

You can use visualize\_input.py to make an input image which will maximize the specific output.
This code is able to maximize a layer's output of any classification model.
(Only dense layer convolutional layer(2D/3D) and pooling layer(2D/3D) are allowed.)

## Requirements
python3  
opencv3 (with ffmpeg), keras, numpy, tqdm  

## Options
Options of 3dcnn.py are as following:  
`--batch`   batch size, default is 128  
`--epoch`   the number of epochs, default is 100  
`--videos`  a name of directory where dataset is stored, default is UCF101  
`--nclass`  the number of classes you want to use, default is 101  
`--output`  a directory where the results described above will be saved  
`--color`   use RGB image or grayscale image, default is False  
`--skip`    get frames at interval or contenuously, default is True  
`--depth`   the number of frames to use, default is 10  

Options of 3dcnn\_ensemble.py are almost same as those of 3dcnn.py.
You can use `--nmodel` option to set the number of models.

Options of visualize\_input.py are as follows:  
`--model` saved json file of a model  
`--weights` saved hd5 file of a model weights  
`--layernames` True to show layer names of a model, default is False  
`--name` the name of a layer which will be maximized  
`--index` the index of a layer output which will be maximized  
`--iter` the number of iteration, default is 20  

You can see more information by using `--help` option
## Demo
You can execute like the following:
```sh
python 3dcnn.py --batch 32 --epoch 50 --videos dataset/ --nclass 10 --output 3dcnnresult/ --color True --skip False --depth 15
```

You can generate the input image which maximizes 0th output of layer named 'dense\_2' like this:
```sh
python visualize_input.py -m result_cnn_10class/ucf101cnnmodel.json -w result_cnn_10class/ucf101cnnmodel.hd5 -n 'dense_2' -i 0 --iter 100
```

When I got the results in [result\_cnn\_10class](https://github.com/rysmarie/MotionRecognition/tree/master/result_cnn_10class), [result\_cnn\_101class](https://github.com/rysmarie/MotionRecognition/tree/master/result_cnn_101class), [result\_3dcnn\_10class](https://github.com/rysmarie/MotionRecognition/tree/master/result_3dcnn_10class), [result\_3dcnn\_101class](https://github.com/rysmarie/MotionRecognition/tree/master/result_3dcnn_101class) , [result\_ensemble](https://github.com/kcct-fujimotolab/3DCNN/tree/master/result_ensemble), I set the options like the follows:

| | nclass | batch | epoch | color | skip | depth | nmodel | accuracy |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|2dcnn.py| 10 | 128 | 100 | False | True | - | - | 0.844 |
|2dcnn.py| 101 | 128 | 100 | False | True | - | - | 0.558 |
|3dcnn.py| 10 | 128 | 100 | False | True | 10 | - | 0.900 |
|3dcnn.py| 101 | 128 | 100 | False | True | 10 | - | 0.692 |
|3dcnn\_ensemble.py| 101 | 128 | 100 | False | True | 10 | 10 | 0.876 |

## Other files
`2dcnn.py`  2DCNN model  
`display.py` get example images from the dataset.  
`videoto3d.py`  get frames from a video, extract a class name from filename of a video in UCF101.  
