/**
 * Extension for MOA EvaluatedPrequentialDelayedCV.java
 *
 * This extension mainly includes:
 *
 * 1. two time window to separately control the delayed labels of instances
 * predicted as positive and negative. More details:
 *
 * In many field, such as fraud detection and software defect prediction,
 * the true label cannot be obtained when the instance come to the data
 * stream. More important,instances predicted as positive will be inspected
 * and labeled manually and relatively fast through the check phone call
 * for fraud detection or SQA inspection for software defect prediction.
 * Therefore, in this extension, a shortWaitingWindow is added to control
 * this procedure. Moreover, instances predicted as negative will be
 * labelled as positive if the true label arrive before a verification
 * latency time window, or labelled as negative if the verification
 * latency time have been consumed. Therefore, in this extension,
 * a longWaitingWindow is added to control this procedure.
 *
 * 2. Finer control of time window units. More details:
 * MOA only supports the number of instances as the parameter of "delayed".
 * However, in real-world, when we ask "how long will it take for the label
 * of the current instance to reach", the answer is usually in terms of time,
 * such as "after one day" or "after two months", rather than in terms of number
 * of instances, such as "after 1000 other instances". Therefore, this extension
 * allow users using the timestamps in their data stream to set time windows.
 *
 * 3. Correct the task procedure to adapt to the real-world scenario. More details:
 * MOA does a task in the following steps:
 *
 * (1) test the instance to obtain the y_pred
 * (2) evaluate the instance by its y_pred and its true label y_true
 * (3) wait for delay and then train the model by its true label and its features
 * Note that for a scenario that the observed true label can only arrive after a delay,
 * you cannot evaluate model by the observed true label before the delay.
 *
 * To make the task work in real situations, the extension works as:
 * (1) test the instance to obtain the y_pred
 * (2) wait for delay to obtain the observed y_true
 * (3) evaluate the instance by its y_pred and y_true
 * (4) train the model by its true label and its features
 *
 *
 * 4. Modify original CV tasks to support detailed evaluation results on each fold.
 *
 * In MOA, tasks whose names contain "CV" provided online distributed evaluation that similar
 * to k-fold validation in offline, which provides a way to support statistic tests.
 * For example, running classifier A and B in a 10-fold cross-validation in MOA
 * with the same random seed, A and B will get 10 pairs of evaluation result at
 * a same time point, such as the following chart.
 * -------------
 *|  A  |   B  |
 * -------------
 * 0.99 | 0.89
 * 0.98 | 0.98
 * 0.96 | 0.96
 * 0.89 | 0.89
 * 0.79 | 0.88
 * 0.88 | 0.87
 * 0.96 | 0.99
 * 0.89 | 0.93
 * 0.96 | 0.92
 * 0.89 | 0.90
 * --------------
 * However, the MOA does not provide the raw evaluation result on each fold, it
 * only provides the averaged performance metrics on all folds in its CV task.
 * Therefore, this extension dump the results on each fold.
 *
 * See details in : my paper citation.
 *
 *
 * @author Xutong Liu (xryu@smail.nju.edu.cn)
 *
 */


 random seed!