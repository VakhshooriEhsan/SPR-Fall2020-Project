# Fisher LDA, Gaussian Mixture Model Classifier, SVM, and Implementation paper related to HMM

## Part A. Fisher LDA (Fisher Linear Discriminant Analysis)

## Part B. Gaussian Mixture Model Classifier

### Results of UKM dataset:

Plot of the training data and test data classified by GMM with K = 1:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.1.1.PNG?raw=true)

Plot of the training data and test data classified by GMM with K = 5:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.1.2.PNG?raw=true)

Plot of the training data and test data classified by GMM with K = 10:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.1.3.PNG?raw=true)

Train and test accuracy and best K:
```
Average train accuracy for each K = 1, 5, 10:
[0.9554289198442851, 0.9914703813141973, 0.9982561793536887]
Average test accuracy for each K = 1, 5, 10:
[0.936500754147813, 0.8504223227752641, 0.7797888386123679]
Best K:
1
```
### Results of iris dataset:

Plot of the training data and test data classified by GMM with K = 1:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.2.1.PNG?raw=true)

Plot of the training data and test data classified by GMM with K = 5:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.2.2.PNG?raw=true)

Plot of the training data and test data classified by GMM with K = 10:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.2.3.PNG?raw=true)

Train and test accuracy and best K:
```
Average train accuracy for each K = 1, 5, 10:
[0.9813333333333335, 0.9943333333333335, 0.9993333333333334]
Average test accuracy for each K = 1, 5, 10:
[0.9746666666666663, 0.9413333333333331, 0.8746666666666667]
Best K:
1
```
### Results of vehicle dataset:

Plot of the training data and test data classified by GMM with K = 1:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.3.1.PNG?raw=true)

Plot of the training data and test data classified by GMM with K = 5:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.3.2.PNG?raw=true)

Plot of the training data and test data classified by GMM with K = 10:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.3.3.PNG?raw=true)
### Results of Health dataset:

Plot of the training data and test data classified by GMM with K = 1:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.4.1.PNG?raw=true)

Plot of the training data and test data classified by GMM with K = 5:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.4.2.PNG?raw=true)

Plot of the training data and test data classified by GMM with K = 10:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/A.4.3.PNG?raw=true)
<!-- ------------------------------------------------------------------------------------------------------------------ -->
## Part C. Support Vector Machine (SVM)
### Results of Linear SVM:
Plot the data and the decision boundary:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/C.1.PNG?raw=true)

train accuracy:
```
accuracy for C=1:
0.975
accuracy for C=100:
1.0

```
### Results of Kernel SVM for two-class:

plots:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/C.2.3.PNG?raw=true)
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/C.2.1.PNG?raw=true)
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/C.2.2.PNG?raw=true)

results:
```
Best C:
10

Best C:
10
Best gamma:
4
Best test accuracy:
0.9469350573023801
```
### Results of Kernel SVM for two-class:

plots:
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/C.3.3.PNG?raw=true)
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/C.3.1.PNG?raw=true)
![representation](https://github.com/VakhshooriEhsan/SPR-Fall2020-Project/blob/master/docs/imgs/C.3.2.PNG?raw=true)
results:
```
Best C:
40

Best C:
10000
Best gamma:
4e-07
Best test accuracy:
0.7384795321637426
```
<!-- ------------------------------------------------------------------------------------------------------------------ -->
## Part D. Implementation Paper Related to HMM

pip install xlrd


Part C:
accuracy for C=1:
1.0
accuracy for C=100:
0.8181818181818182

[[0.53032618 0.5304496  0.53702028 0.53037908 0.53146635 0.71120188
  0.54843961 0.53040846]
 [0.53043197 0.53044373 0.53043785 0.81409345 0.83598002 0.88367323
  0.90264473 0.86420217]
 [0.53039083 0.53043197 0.70516603 0.84816339 0.88124008 0.90461358
  0.91336468 0.93482809]
 [0.53040259 0.8228563  0.8554687  0.89879518 0.92115192 0.92701146
  0.93867764 0.93723185]
 [0.71583309 0.86325595 0.8929474  0.92160447 0.92943873 0.93672054
  0.93769615 0.93382897]
 [0.86569498 0.90411989 0.9221099  0.92654129 0.92699383 0.93816632
  0.93720835 0.92940934]
 [0.89682045 0.92411989 0.92259183 0.92940347 0.9353159  0.94792242
  0.93483397 0.9314017 ]
 [0.91727299 0.9269762  0.92702909 0.92993241 0.93238907 0.94256244
  0.9343109  0.92989127]]
Best C:
10
Best gamma:
4

[[0.55192982 0.44900585 0.29649123 0.23625731 0.23649123 0.23637427
  0.24070175 0.24701754]
 [0.55777778 0.43812865 0.27614035 0.23625731 0.24292398 0.24280702
  0.22432749 0.24912281]
 [0.54315789 0.44233918 0.30035088 0.2451462  0.21298246 0.22783626
  0.22994152 0.25274854]
 [0.5625731  0.42409357 0.2805848  0.24023392 0.24678363 0.25976608
  0.23590643 0.24011696]
 [0.57649123 0.4302924  0.33052632 0.24023392 0.2154386  0.22959064
  0.19953216 0.25789474]
 [0.60292398 0.49157895 0.2954386  0.25099415 0.23988304 0.20654971
  0.23426901 0.24304094]
 [0.59719298 0.46736842 0.30982456 0.23391813 0.25707602 0.24023392
  0.26128655 0.26      ]
 [0.57064327 0.4805848  0.32093567 0.24912281 0.24222222 0.21707602
  0.25157895 0.22093567]]
Best C:
4
Best gamma:
0.001
