# A Hybrid Spiking Recurrent Neural Network on Hardware for Efficient Emotion Recognition

***
**This code can be used as the supplemental material for the paper: "A Hybrid Spiking Recurrent Neural Network on Hardware for Efficient Emotion Recognition". (Published on *IEEE AICAS*, June, 2022)**.
***

## Citation:
C. Zou, X. Cui, Y. Kuang, Y. Wang,  and X. Wang, "A Hybrid Spiking Recurrent Neural Network on Hardware for Efficient Emotion Recognition," 2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS), 2022, pp. 1-4, doi: xxx.

### **Features**:
- This supplemental material gives a reproduction function of training and testing experiments of vanillaRNN, LSTM, Text-CNN and proposed spiking RNN (SRNN) in our paper. Two kinds of emotion recognition datasets with different sentence lengths (also in English and Chinese language) are considered.


## File overview:
- `README.md` - this readme file.<br>
- `Cars` - the workspace folder for `networks` on the Car dataset with `time step = 30`.<br>
- `Movies` - the workspace folder for `networks` on the Movie dataset with `time step = 500`.<br>

## Requirements
### **Dependencies and Libraries**:
* python 3.5 (https://www.python.org/ or https://www.anaconda.com/)
* pytorch 0.4.1 (https://pytorch.org/)
* torchtext 0.3.1, glove.6B
* CPU: Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
* GPU: Tesla V100

### **Datasets**:
* Movies: [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), [preprocessing](https://blog.csdn.net/weixin_42479155/article/details/104491750), [reference](https://blog.csdn.net/qq_30057549/article/details/103225576)
* Cars: [dataset](https://github.com/WHLYA/text-classification/tree/master/text%20classification), 
[preprocessing](https://blog.csdn.net/qsmx666/article/details/105648175), [reference](https://blog.csdn.net/u014514939/article/details/88834548)

## **Run the code**:
for example (networks training and testing, *Movies dataset*):
```sh
$ cd Movies
$ python movies_main.py
$ python text_cnn.py
```

for example (networks training and testing, *Cars dataset*):
```sh
$ cd Cars
$ python cars_main.py
$ python text_cnn.py
```

## Others
* You can run the `plot_sop.py` and `plot_sparsity.py` to get illustration information.

## Results
Please refer to our paper for more information.

## More question:<br>
- There might be a little difference of results for multiple training repetitions, because of the randomization. 
- Please feel free to reach out here or email: 1801111301@pku.edu.cn, if you have any questions or difficulties. I'm happy to help guide you.

