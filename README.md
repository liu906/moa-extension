# MOA-extension

A extension of MOA framework (based on the version of source code from MOA github https://github.com/waikato/moa that last updated at 2022/11/28 )

## Extension of EvaluatePrequentialDelayedCV
Based on EvaluatePrequentialDelayedCV class, we add three new java files for delayed cv prequential evaluation:    

1. EvaluatePrequentialDelayedCVExtension
2. EvaluatePrequentialDelayedCVIdeal
3. EvaluatePrequentialDelayedCVPosNegWindow
### A. new features of delayed cv prequential evaluation
Those three java classes have the following new features:
#### 1.two time window to separately control the delayed labels of instances predicted as positive and negative. 
In many field, such as fraud detection and software defect prediction, the true label cannot be obtained when the instance come to the data stream. More important,instances predicted as positive will be inspected and labeled manually and relatively fast through the check phone call for fraud detection or SQA inspection for software defect prediction. Therefore, in this extension, a shortWaitingWindow is added to control this procedure. Moreover, instances predicted as negative will be labelled as positive if the true label arrive before a verification latency time window, or labelled as negative if the verification latency time have been consumed. Therefore, in this extension, a longWaitingWindow is added to control this procedure.

#### 2.Finer control of time window units.
MOA only supports the number of instances as the parameter of "delayed". However, in real-world, when we ask "how long will it take for the label of the current instance to reach", the answer is usually in terms of time, such as "after one day" or "after two months", rather than in terms of number of instances, such as "after 1000 other instances". Therefore, this extension allow users using the timestamps in their data stream to set time windows.
 
 #### 3. Modify the task procedure to adapt to the real-world scenario. 
  MOA does a task in the following steps:  
  (1) test the instance to obtain the y_pred
  (2) evaluate the instance by its y_pred and its true label y_true  
  (3) wait for delay and then train the model by its true label and its features  
 
 Note that for a scenario that the observed true label can only arrive after a delay, you cannot evaluate model by the observed true label before the delay.  


 To make the task work in real situations, the extension works as:  
  (1) test the instance to obtain the y_pred  
  (2) wait for delay to obtain the observed y_true  
  (3) evaluate the instance by its y_pred and y_true  
  (4) train the model by its true label and its features  
 
#### 4. Support detailed evaluation results on each fold  

In MOA, tasks whose names contain "CV" provided online distributed evaluation that similar to k-fold validation in offline, which provides a way to support statistic tests. For example, running classifier A and B in a 10-fold cross-validation in MOA with the same random seed, A and B will get 10 pairs of evaluation result at a same time point, such as the following chart.
 
|A| B |
|--|--|
|0.99 | 0.89|
|0.98 | 0.98|
|0.96 | 0.96|
|0.89 | 0.89|
|0.79 | 0.88|
|0.88 | 0.87|
|0.96 | 0.99|
|0.89 | 0.93|
|0.96 | 0.92|
|0.89 | 0.90|
 However, the MOA does not provide the raw evaluation result on each fold, it only provides the averaged performance metrics on all folds in its CV task. Therefore, this extension dump the results on each fold.  


### B. add a new parameter to control random seed of online classifiers 
The MOA framework (until 2023/3/17) only provided control of random seed for stream validation before, although the have desiged several classifiers with randomness, their random seeds cannot be controled directly through a parameter in command line running. Therefore, I added a parameter called **'-x'** to control the random seed of bootstrap validation, and parameter **'-r'** is changed to control the random seed of classifier with randomness now.

## Explanation of three extended evaluation procedures
### EvaluatePrequentialDelayedCVExtension
The delay time of observed label arriving for instances predicted as positive and instances predicted as negative are the same. For example,

    java -classpath "classes" moa.DoTask EvaluatePrequentialDelayedCVExtension -l trees.HoeffdingTree -s "(ArffFileStream" -f "brackets.arff)" -e "(FadingFactorClassificationPerformanceEvaluator" -a 0.99  -n 0 -o -p -r "-f)" -k 99 -f 30 -q 30 -d "dumpFile.csv" -o "detail.csv" -a Bootstrap-Validation -D 0 -w 10 -A 1 -r 1 -P 86400 -N 86400

### EvaluatePrequentialDelayedCVIdeal
Ideal situation of cv delayed prequential evaluation, which means of observed label arriving is 0. For example,
 

    java -classpath "classes" moa.DoTask EvaluatePrequentialDelayedCVIdeal -l trees.HoeffdingTree -s "(ArffFileStream" -f "brackets.arff)" -e "(FadingFactorClassificationPerformanceEvaluator" -a 0.99  -n 0 -o -p -r "-f)" -k 99 -f 30 -q 30 -d "dumpFile.csv" -o "detail.csv" -a Bootstrap-Validation -D 0 -w 10 -A 1 -r 1

 
### EvaluatePrequentialDelayedCVPosNegWindow
The delay time of observed label arriving for instances predicted as positive and instances predicted as negative are not the same. For example,

    java -classpath "classes" moa.DoTask EvaluatePrequentialDelayedCVPosNegWindow -l trees.HoeffdingTree -s "(ArffFileStream" -f "brackets.arff)" -e "(FadingFactorClassificationPerformanceEvaluator" -a 0.99  -n 0 -o -p -r "-f)" -k 99 -f 30 -q 30 -d "dumpFile.csv" -o "detail.csv" -a Bootstrap-Validation -D 0 -w 10 -A 1 -r 1 -P 86400 -N 1296000

