# Fisher LDA, Gaussian Mixture Model Classifier, SVM, and Implementation paper related to HMM

## Part A. Fisher LDA (Fisher Linear Discriminant Analysis)

Implement of Fisher LDA.
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

Implement of `Forward`, `Backward`, `Viterbi`, `Baum welch` algorithms.
