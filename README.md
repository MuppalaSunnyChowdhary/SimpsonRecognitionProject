# SimpsonRecognitionProject

Training a Convolutional Neural Network to recognize The Simpson TV Show characters using Keras (TensorFlow backend).  

### Part 0 : Collecting data

This part is about collecting and labeling Simpson pictures.  
Most of the pictures are from Simpson video, analyzed frame by frame.

Run ``python3 label_data.py`` into a folder with Simpson episodes (.avi format) to analyze them and label frames.  
You crop each frame (left part, right part, full-frame, nothing) and then label it.  

You can find the dataset on [Kaggle](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)

### Part 1 : Training with Keras

I aim to have 1000 pictures per class (for 20 classes), unfortunately some characters are not often on screen so I have fewer pictures for those characters.
As you can see on the Jupyter notebook, I benchmark two models : 4 and 6 convolutional layers neural networks. Because of the small number of pictures (approx. 1k pictures per class), I use data augmentation.  
Currently, I have 96% of accuracy (F1-Score) for 18 classes.  

### Part 2 : Faster R-CNN

The second part is about upgrading the deep learning model to detect and recognize characters. I have to annotate data to get bounding boxes for characters for each picture in order to train a new model : [Faster R-CNN](https://arxiv.org/abs/1506.01497) (which is based on a Region Proposal Network).


### Files description

1.  `label_data.py` : tools functions for notebooks + script to name characters from frames from .avi videos  
2.  `label_pointer.py` : point with mouse clicks to save bounding box coordinates on annotations text file (from already labeled pictures)
3.  `train.py` : training simple convnet
4.  `train_frcnn.py -p annotation.txt` : training Faster R-CNN with data from the annotation text file
5.  `test_frcnn.py -p path/test_data/` : testing Faster R-CNN 

![Lisa picture](https://github.com/alexattia/SimpsonRecognition/blob/master/pics/mapple_lisa.png)
