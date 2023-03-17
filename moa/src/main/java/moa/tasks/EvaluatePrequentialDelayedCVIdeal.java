/*
 *    EvaluatePrequentialDelayedCV.java
 *
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.tasks;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import moa.classifiers.MultiClassClassifier;
import moa.core.*;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.evaluation.preview.LearningCurveExtension;
import moa.learners.Learner;
import moa.options.ClassOption;
import moa.streams.ExampleStream;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

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
public class EvaluatePrequentialDelayedCVIdeal extends ClassificationMainTask {


    @Override
    public String getPurposeString() {
        return "Evaluates a classifier using delayed cross-validation evaluation "
                + "by testing and only training with the example after the arrival of other k examples (delayed labeling) ";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Learner to train.", MultiClassClassifier.class, "moa.classifiers.bayes.NaiveBayes");

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", ExampleStream.class,
            "generators.RandomTreeGenerator");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            LearningPerformanceEvaluator.class,
            "WindowClassificationPerformanceEvaluator");

    public IntOption delayLengthOption = new IntOption("delay", 'k',
            "Number of instances before test instance is used for training",
            1000, 1, Integer.MAX_VALUE);

    public IntOption instanceLimitOption = new IntOption("instanceLimit", 'i',
            "Maximum number of instances to test/train on  (-1 = no limit).",
            100000000, -1, Integer.MAX_VALUE);

    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of seconds to test/train for (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",
            'f',
            "How many instances between samples of the learning performance.",
            100000, 0, Integer.MAX_VALUE);

    public IntOption memCheckFrequencyOption = new IntOption(
            "memCheckFrequency", 'q',
            "How many instances between memory bound checks.", 100000, 0,
            Integer.MAX_VALUE);

    public IntOption dateIndexOption = new IntOption(
            "dateIndex", 'D',
            "date index in the data stream. -1 means no date column in the stream. -1 as default", -1, -1,
            Integer.MAX_VALUE);

    public IntOption feedbackIndexOption = new IntOption(
            "feedbackIndex", 'A',
            "feedback index in the data stream. note that feedback column will be deleted before training. " +
                    "-1 means no date column in the stream. -1 as default", -1, -1,
            Integer.MAX_VALUE);
    public IntOption positiveFeedBackTimeOption = new IntOption(
            "positiveFeedBackTimeLimit", 'P',
            "how long between instances be predicted as positive and get their observed labels", 0, 0,
            Integer.MAX_VALUE);
    public IntOption negativeFeedBackTimeOption = new IntOption(
            "negativeFeedBackTimeLimit", 'N',
            "how long between instances be predicted as negative and get their observed labels", 0, 0,
            Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public FileOption dumpFoldFileOption = new FileOption("dumpFoldFile", 'o',
            "File to append intermediate csv results on each fold to.", null, "csv", true);

    public IntOption numFoldsOption = new IntOption("numFolds", 'w',
            "The number of folds (e.g. distributed models) to be used.", 10, 1, Integer.MAX_VALUE);

    public MultiChoiceOption validationMethodologyOption = new MultiChoiceOption(
            "validationMethodology", 'a', "Validation methodology to use.", new String[]{
            "Cross-Validation", "Bootstrap-Validation", "Split-Validation"},
            new String[]{"k-fold distributed Cross Validation",
                    "k-fold distributed Bootstrap Validation",
                    "k-fold distributed Split Validation"
            }, 0);

    public IntOption randomSeedOption = new IntOption("randomSeed", 'r',
            "Seed for random behaviour of the task.", 1);
    public IntOption bvRandomSeedOption = new IntOption("bootStrapValidationRandomSeed", 'x',
            "Seed for random behaviour of the task.", 1);
    protected int positiveClass = 1;
    protected int negativeClass = 0;

    // Buffer of instances to use for training.
    // Note: It is a list of lists because it stores instances per learner, e.g.
    // CV of 10, would be 10 lists of buffered instances for delayed training.
    protected LinkedList<LinkedList<Example>> trainInstances;
    protected LinkedList<LinkedList<Example>> positiveTrainInstances;
    protected LinkedList<LinkedList<Example>> negativeTrainInstances;
    protected LinkedList<LinkedList<String>> trainTimestamps;
    protected LinkedList<LinkedList<String>> positiveTrainTimestamps;
    protected LinkedList<LinkedList<String>> negativeTrainTimestamps;
    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }
    public static boolean firstFoldDump = true;
    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
        int feedbackIndex= this.feedbackIndexOption.getValue();
        Random random = new Random(this.bvRandomSeedOption.getValue());
        ExampleStream stream = (ExampleStream) getPreparedClassOption(this.streamOption);

        Learner[] learners = new Learner[this.numFoldsOption.getValue()];
        Learner baseLearner = (Learner) getPreparedClassOption(this.learnerOption);
        if (baseLearner.isRandomizable()) {
            baseLearner.setRandomSeed(this.randomSeedOption.getValue());
            baseLearner.resetLearning();
        }
//        baseLearner.resetLearning();

        /* evaluating */
        LearningPerformanceEvaluator[] evaluators = new LearningPerformanceEvaluator[this.numFoldsOption.getValue()];
        LearningPerformanceEvaluator baseEvaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        int[] arrInstancesTested = new int[learners.length];
        for (int i = 0; i < learners.length; i++) {
            learners[i] = (Learner) baseLearner.copy();
            learners[i].setModelContext(stream.getHeader());
            evaluators[i] = (LearningPerformanceEvaluator) baseEvaluator.copy();
            arrInstancesTested[i] = 0;
        }


        LearningCurve learningCurve = new LearningCurve(
                "learning evaluation instances");

        LearningCurveExtension learningFoldCurve = new LearningCurveExtension(
                "learning evaluation instances on certain fold");

        int maxInstances = this.instanceLimitOption.getValue();
        long instancesProcessed = 0;
        int maxSeconds = this.timeLimitOption.getValue();
        int secondsElapsed = 0;
        monitor.setCurrentActivity("Evaluating learner...", -1.0);

        int dateIndex = this.dateIndexOption.getValue();

        this.trainInstances = new LinkedList<LinkedList<Example>>();
        this.positiveTrainInstances = new LinkedList<LinkedList<Example>>();
        this.negativeTrainInstances = new LinkedList<LinkedList<Example>>();
        this.trainTimestamps = new LinkedList<LinkedList<String>>();
        this.positiveTrainTimestamps = new LinkedList<LinkedList<String>>();
        this.negativeTrainTimestamps = new LinkedList<LinkedList<String>>();


        for(int i = 0; i < learners.length; i++) {
            this.trainInstances.add(new LinkedList<Example>());
            this.positiveTrainInstances.add(new LinkedList<Example>());
            this.negativeTrainInstances.add(new LinkedList<Example>());

            this.trainTimestamps.add(new LinkedList<String>());
            this.positiveTrainTimestamps.add(new LinkedList<String>());
            this.negativeTrainTimestamps.add(new LinkedList<String>());
        }

        File dumpFile = this.dumpFileOption.getFile();
        File dumpFoldFile = this.dumpFoldFileOption.getFile();

        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }

        boolean firstDump = true;
        PrintStream immediateFoldResultStream = null;
        if (dumpFoldFile != null) {
            try {
                if (dumpFoldFile.exists()) {
                    immediateFoldResultStream = new PrintStream(
                            new FileOutputStream(dumpFoldFile, true), true);
                } else {
                    immediateFoldResultStream = new PrintStream(
                            new FileOutputStream(dumpFoldFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFoldFile, ex);
            }
        }
        boolean firstFoldDump = true;


        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        long lastEvaluateStartTime = evaluateStartTime;
        double RAMHours = 0.0;

        while (stream.hasMoreInstances()
                && ((maxInstances < 0) || (instancesProcessed < maxInstances))
                && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {


            Example trainInst = stream.nextInstance();

            // TODO: newly added to be check
            instancesProcessed++;

            String trainInstTimestamp = ((InstanceExample) trainInst).instance.stringValue(dateIndex);
            if(dateIndex != dateIndexOption.getMinValue()){
                ((InstanceImpl) ((InstanceExample) trainInst).instance).instanceHeader.getInstanceInformation().deleteAttributeAt(dateIndex);
                ((InstanceExample)trainInst).getData().deleteAttributeAt(dateIndex);
            }

            String feedbackValue = ((InstanceExample) trainInst).instance.stringValue(feedbackIndex-1);
            if(feedbackIndex != feedbackIndexOption.getMinValue()){
                ((InstanceImpl) ((InstanceExample) trainInst).instance).instanceHeader.getInstanceInformation().deleteAttributeAt(feedbackIndex-1);
                ((InstanceExample)trainInst).getData().deleteAttributeAt(feedbackIndex);
            }
            //Example testInst = (Example) trainInst;


            //分配实例给每个fold
            //test it 拿到它的predicted label
            //if predicted label==positive then positiveQueue.add(instance) else negativeQueue.add(instances)
            //如果到时间了就从positiveInstances队列取出来一个赋予他observed label
            //然后立马evaluated
            //随后train by it
            boolean newEvaluated = false;

            for (int i = 0; i < learners.length; i++) {
                //分配实例给每个fold
                int k = 1;
                switch (this.validationMethodologyOption.getChosenIndex()) {
                    // TODO: the update of instancesProcessed may have bug. therefore k is always 1 in case 0 and case 2
                    case 0: //Cross-Validation;
                        k = instancesProcessed % learners.length == i ? 0: 1; //Test all except one
                        break;
                    case 1: //Bootstrap;
                        k = MiscUtils.poisson(1, random);
                        break;
                    case 2: //Split-Validation;
                        k = instancesProcessed % learners.length == i ? 1: 0; //Test only one
                        break;
                }

                //test it 拿到它的predicted label
                double[] prediction = learners[i].getVotesForInstance(trainInst);
                int predictedClass = Utils.maxIndex(prediction);


                //if predicted label==positive then positiveQueue.add(instance) else negativeQueue.add(instances)
                if (k > 0) {
                    if(predictedClass==positiveClass){
                        this.positiveTrainInstances.get(i).addLast(trainInst);
                        this.positiveTrainTimestamps.get(i).addLast(trainInstTimestamp);
                    }else{
                        this.negativeTrainInstances.get(i).addLast(trainInst);
                        this.negativeTrainTimestamps.get(i).addLast(trainInstTimestamp);
                    }
                }
                boolean isEvaluated = false;

                //如果时间不到但是feedback instance到了. in this case, observed label is true label
                int indexOfLabelledPosInstance = this.positiveTrainTimestamps.get(i).indexOf(feedbackValue);
                int indexOfLabelledNegInstance = this.negativeTrainTimestamps.get(i).indexOf(feedbackValue);
                if(Math.max(indexOfLabelledNegInstance,indexOfLabelledPosInstance)!=-1){
                    if(indexOfLabelledPosInstance!=-1){
                        int observedLabel = stream.getHeader().numClasses();
                        isEvaluated = true;
                        Example trainInstI = this.positiveTrainInstances.get(i).get(indexOfLabelledPosInstance);
                        this.positiveTrainInstances.get(i).remove(indexOfLabelledPosInstance);
                        this.positiveTrainTimestamps.get(i).remove(indexOfLabelledPosInstance);
                        evaluators[i].addResult(trainInstI, prediction);
                        learners[i].trainOnInstance(trainInstI);
                        arrInstancesTested[i]++;
                        addEvaluationOnFoldLevel(arrInstancesTested,i,evaluateStartTime,lastEvaluateStartTime,
                                learners,RAMHours,learningFoldCurve,preciseCPUTiming,evaluators,trainInstTimestamp,
                                immediateFoldResultStream);


                    }else if(indexOfLabelledNegInstance!=-1){
                        isEvaluated = true;
                        Example trainInstI = this.negativeTrainInstances.get(i).get(indexOfLabelledNegInstance);
                        this.negativeTrainInstances.get(i).remove(indexOfLabelledNegInstance);
                        this.negativeTrainTimestamps.get(i).remove(indexOfLabelledNegInstance);
                        evaluators[i].addResult(trainInstI, prediction);
                        learners[i].trainOnInstance(trainInstI);
                        arrInstancesTested[i]++;
                        addEvaluationOnFoldLevel(arrInstancesTested,i,evaluateStartTime,lastEvaluateStartTime,
                                learners,RAMHours,learningFoldCurve,preciseCPUTiming,evaluators,trainInstTimestamp,
                                immediateFoldResultStream);

                    }
                }

                /* 到时间了就从positiveInstances队列取出来一个赋予他observed label. in this case, observed label is negative
                然后立马evaluated
                随后train by it*/
                if (this.positiveTrainTimestamps.get(i).size() != 0 &&
                        this.positiveFeedBackTimeOption.getValue() <=
                                (Integer.valueOf(trainInstTimestamp) - Integer.valueOf(this.positiveTrainTimestamps.get(i).getFirst()))) {//把.size改成.timestamp是不是就可以实现QAtimeWindow了
                    isEvaluated = true;
                    Example trainInstI = this.positiveTrainInstances.get(i).removeFirst();
                    this.positiveTrainTimestamps.get(i).removeFirst();
                    evaluators[i].addResult(trainInstI, prediction);//原本的evaluators 里面的实例的到达顺序会被我的positive和negative窗口的加入打乱默认的先进先出的顺序
                    learners[i].trainOnInstance(trainInstI);
                    arrInstancesTested[i]++;
                    addEvaluationOnFoldLevel(arrInstancesTested,i,evaluateStartTime,lastEvaluateStartTime,
                            learners,RAMHours,learningFoldCurve,preciseCPUTiming,evaluators,trainInstTimestamp,
                            immediateFoldResultStream);

                }

                if(this.negativeTrainTimestamps.get(i).size() != 0 &&
                        this.negativeFeedBackTimeOption.getValue() <=
                                (Integer.valueOf(trainInstTimestamp)  - Integer.valueOf(this.negativeTrainTimestamps.get(i).getFirst()))) {//把.size改成.timestamp是不是就可以实现QAtimeWindow了
                    isEvaluated = true;
                    Example trainInstI = this.negativeTrainInstances.get(i).removeFirst();
                    this.negativeTrainTimestamps.get(i).removeFirst();
                    evaluators[i].addResult(trainInstI, prediction);
                    learners[i].trainOnInstance(trainInstI);
                    arrInstancesTested[i]++;
                    addEvaluationOnFoldLevel(arrInstancesTested,i,evaluateStartTime,lastEvaluateStartTime,
                            learners,RAMHours,learningFoldCurve,preciseCPUTiming,evaluators,trainInstTimestamp,
                            immediateFoldResultStream);

                }

                if(isEvaluated && !newEvaluated){
                    newEvaluated = true;
                }
            }

            if (newEvaluated && (instancesProcessed % this.sampleFrequencyOption.getValue() == 0
                    || stream.hasMoreInstances() == false)) {
                long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);

                for (int i = 0; i < learners.length; i++) {
                    double RAMHoursIncrement = learners[i].measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                    RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
                    RAMHours += RAMHoursIncrement;
                }

                lastEvaluateStartTime = evaluateTime;
                learningCurve.insertEntry(new LearningEvaluation(
                        getEvaluationMeasurements(
                                new Measurement[]{
                                        new Measurement(
                                                "learning evaluation instances",
                                                instancesProcessed),
                                        new Measurement(
                                                "evaluation time ("
                                                        + (preciseCPUTiming ? "cpu "
                                                        : "") + "seconds)",
                                                time),
                                        new Measurement(
                                                "model cost (RAM-Hours)",
                                                RAMHours)
                                }, evaluators)));

                if (immediateResultStream != null) {
                    if (firstDump) {
                        immediateResultStream.println(learningCurve.headerToString());
                        firstDump = false;
                    }
                    immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                    immediateResultStream.flush();
                }
            }

            if (instancesProcessed != 0 && instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                if (monitor.taskShouldAbort()) {
                    return null;
                }
                long estimatedRemainingInstances = stream.estimatedRemainingInstances();
                if (maxInstances > 0) {
                    long maxRemaining = maxInstances - instancesProcessed;
                    if ((estimatedRemainingInstances < 0)
                            || (maxRemaining < estimatedRemainingInstances)) {
                        estimatedRemainingInstances = maxRemaining;
                    }
                }
                monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
                        : (double) instancesProcessed
                        / (double) (instancesProcessed + estimatedRemainingInstances));
                if (monitor.resultPreviewRequested()) {
                    monitor.setLatestResultPreview(learningCurve.copy());
                }
                secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()
                        - evaluateStartTime);
            }
        }
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }

        if (immediateFoldResultStream != null) {
            immediateFoldResultStream.close();
        }

        return learningCurve;

    }
    public void addEvaluationOnFoldLevel(int[] arrInstancesTested,int i,long evaluateStartTime,long lastEvaluateStartTime,
                                         Learner[] learners,double RAMHours,LearningCurveExtension learningFoldCurve,
                                         boolean preciseCPUTiming,LearningPerformanceEvaluator[] evaluators,String trainInstTimestamp,
                                         PrintStream immediateFoldResultStream){
        if (arrInstancesTested[i]!=0) {
            long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
            double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
            double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);


            double RAMHoursIncrement = learners[i].measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
            RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
            RAMHours += RAMHoursIncrement;

            if (arrInstancesTested[i] % this.sampleFrequencyOption.getValue() == 0){
                learningFoldCurve.insertEntry(new LearningEvaluation(
                        getFoldEvaluationMeasurements(
                                new Measurement[]{
                                        new Measurement(
                                                "learning evaluation instances on certain fold",
                                                arrInstancesTested[i]),
                                        new Measurement(
                                                "evaluation time ("
                                                        + (preciseCPUTiming ? "cpu "
                                                        : "") + "seconds)",
                                                time),
                                        new Measurement(
                                                "model cost (RAM-Hours)",
                                                RAMHours)
                                }, evaluators[i], i, trainInstTimestamp)));

                if (immediateFoldResultStream != null) {
                    if (firstFoldDump) {
                        immediateFoldResultStream.println(learningFoldCurve.headerToString());
                        firstFoldDump = false;
                    }
                    immediateFoldResultStream.println(learningFoldCurve.entryToString(learningFoldCurve.numEntries() - 1));
                    immediateFoldResultStream.flush();
                }

            }

        }
    }

    public Measurement[] getEvaluationMeasurements(Measurement[] modelMeasurements, LearningPerformanceEvaluator[] subEvaluators) {
        List<Measurement> measurementList = new LinkedList<>();
        if (modelMeasurements != null) {
            measurementList.addAll(Arrays.asList(modelMeasurements));
        }
        // add average of sub-model measurements
        if ((subEvaluators != null) && (subEvaluators.length > 0)) {
            List<Measurement[]> subMeasurements = new LinkedList<>();
            for (LearningPerformanceEvaluator subEvaluator : subEvaluators) {
                if (((BasicClassificationPerformanceEvaluator) subEvaluator).numClasses != 0) {
                    subMeasurements.add(subEvaluator.getPerformanceMeasurements());
                }
            }
            Measurement[] avgMeasurements = Measurement.averageMeasurements(subMeasurements.toArray(new Measurement[subMeasurements.size()][]));
            measurementList.addAll(Arrays.asList(avgMeasurements));
        }
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }

    public Measurement[] getFoldEvaluationMeasurements(Measurement[] modelMeasurements, LearningPerformanceEvaluator subEvaluator, int fold, String timestamp) {
        List<Measurement> measurementList = new LinkedList<>();
        measurementList.add(new Measurement("current timestamp",Double.valueOf(timestamp).longValue()));
        measurementList.add(new Measurement("fold",fold));
        if (modelMeasurements != null) {
            measurementList.addAll(Arrays.asList(modelMeasurements));
        }
        List<Measurement[]> subMeasurements = new LinkedList<>();

        if (subEvaluator != null) {
            subMeasurements.add(subEvaluator.getPerformanceMeasurements());
            // Measurement[] avgMeasurements = Measurement.averageMeasurements(subMeasurements.toArray(new Measurement[subMeasurements.size()][]));

            measurementList.addAll(Arrays.asList(subMeasurements.get(0)));
        }

        return measurementList.toArray(new Measurement[measurementList.size()]);
    }
}
