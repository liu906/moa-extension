/*
 *    BasicClassificationPerformanceEvaluator.java
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
package moa.evaluation;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.core.*;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Prediction;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map;

/**
 * Classification evaluator that performs basic incremental evaluation.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * 
 * Updates in September 15th 2017 to include precision, recall and F1 scores.
 * @author Jean Karax (karaxjr@gmail.com)
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @author Wilson Sasaki Jr (sasaki.wilson.jr@gmail.com)
 * @version $Revision: 8 $
 */
public class BasicClassificationPerformanceEvaluator extends AbstractOptionHandler
        implements ClassificationPerformanceEvaluator {

    private static final long serialVersionUID = 1L;

    protected Estimator weightCorrect;

    protected Estimator[] columnKappa;

    protected Estimator[] rowKappa;

    protected Estimator[] precision;

    protected Estimator[] recall;
    protected Estimator[] recall_weightCorrectNoChangeClassifier;
    protected Estimator[] recall_weightMajorityClassifier;
    protected Estimator[] precision_weightCorrectNoChangeClassifier;
    protected Estimator[] precision_weightMajorityClassifier;

    public int numClasses;

    private Estimator weightCorrectNoChangeClassifier;

    private Estimator weightMajorityClassifier;

    private int lastSeenClass;

    private double totalWeightObserved;

    public FlagOption precisionRecallOutputOption = new FlagOption("precisionRecallOutput",
            'o',
            "Outputs average precision, recall and F1 scores.");
    
    public FlagOption precisionPerClassOption = new FlagOption("precisionPerClass",
            'p',
            "Report precision per class.");

    public FlagOption recallPerClassOption = new FlagOption("recallPerClass",
            'r',
            "Report recall per class.");

    public FlagOption f1PerClassOption = new FlagOption("f1PerClass", 'f',
            "Report F1 per class.");

    public FloatOption noiseOption = new FloatOption("noise",'n',"add noise filter to the prediction result",
            0,0,1);

    @Override
    public void reset() {
        reset(this.numClasses);
    }

    public void reset(int numClasses) {
        this.numClasses = numClasses;
        this.rowKappa = new Estimator[numClasses];
        this.columnKappa = new Estimator[numClasses];
        this.precision = new Estimator[numClasses];
        this.recall = new Estimator[numClasses];
        this.recall_weightMajorityClassifier = new Estimator[numClasses];
        this.recall_weightCorrectNoChangeClassifier = new Estimator[numClasses];
        this.precision_weightMajorityClassifier = new Estimator[numClasses];
        this.precision_weightCorrectNoChangeClassifier = new Estimator[numClasses];

        for (int i = 0; i < this.numClasses; i++) {
            this.rowKappa[i] = newEstimator();
            this.columnKappa[i] = newEstimator();
            this.precision[i] = newEstimator();
            this.recall[i] = newEstimator();
            this.recall_weightCorrectNoChangeClassifier[i] = newEstimator();
            this.recall_weightMajorityClassifier[i] = newEstimator();
            this.precision_weightCorrectNoChangeClassifier[i] = newEstimator();
            this.precision_weightMajorityClassifier[i] = newEstimator();
        }
        this.weightCorrect = newEstimator();
        this.weightCorrectNoChangeClassifier = newEstimator();
        this.weightMajorityClassifier = newEstimator();
        this.lastSeenClass = 0;
        this.totalWeightObserved = 0;
    }

    @Override
    public void addResult(Example<Instance> example, double[] classVotes) {
        Instance inst = example.getData();
        double weight = inst.weight();
        if (inst.classIsMissing() == false) {
            int trueClass = (int) inst.classValue();
            int predictedClass = Utils.maxIndex(classVotes);
            double noise = noiseOption.getValue();
            if (Math.random()<=noise){
              // Map<String, Integer> valuesStringAttribute = ((InstanceImpl) ((InstanceExample) example).instance).instanceHeader.getInstanceInformation().attributesInformation.attribute(((InstanceExample) example).instance.classIndex()).valuesStringAttribute;
                predictedClass = predictedClass==0?1:0;
            }
            if (weight > 0.0) {
                if (this.totalWeightObserved == 0) {
                    reset(inst.dataset().numClasses());
                }
                this.totalWeightObserved += weight;
                this.weightCorrect.add(predictedClass == trueClass ? weight : 0);
                for (int i = 0; i < this.numClasses; i++) {
                    this.rowKappa[i].add(predictedClass == i ? weight : 0);
                    this.columnKappa[i].add(trueClass == i ? weight : 0);
                    // for both precision and recall, NaN values are used to 'balance' the number
                    // of instances seen across classes
                    // Xutong Liu: but those NaN values cause estimator function get NaN result under AdwinClassificationPerformanceEvaluator
                    // How can I fix it?
                    if (predictedClass == i) {
                        precision[i].add(predictedClass == trueClass ? weight : 0.0);
                        precision_weightMajorityClassifier[i].add(getMajorityClass()==trueClass ? weight : 0.0);
                        precision_weightCorrectNoChangeClassifier[i].add(this.lastSeenClass == trueClass ? weight : 0.0);
                    } else {
                        precision[i].add(Double.NaN);
                        precision_weightMajorityClassifier[i].add(Double.NaN);
                        precision_weightCorrectNoChangeClassifier[i].add(Double.NaN);
                    }

                    if (trueClass == i) {
                        recall[i].add(predictedClass == trueClass ? weight : 0.0);
                        recall_weightCorrectNoChangeClassifier[i].add(this.lastSeenClass == trueClass ? weight : 0.0);
                        recall_weightMajorityClassifier[i].add(getMajorityClass() == trueClass ? weight : 0.0);
                    } else {
                        recall[i].add(Double.NaN);
                        recall_weightMajorityClassifier[i].add(Double.NaN);
                        recall_weightCorrectNoChangeClassifier[i].add(Double.NaN);
                    }
                }
            }
            this.weightCorrectNoChangeClassifier.add(this.lastSeenClass == trueClass ? weight : 0);
            this.weightMajorityClassifier.add(getMajorityClass() == trueClass ? weight : 0);
            this.lastSeenClass = trueClass;
        }
    }

    private int getMajorityClass() {
        int majorityClass = 0;
        double maxProbClass = 0.0;
        for (int i = 0; i < this.numClasses; i++) {
            if (this.columnKappa[i].estimation() > maxProbClass) {
                majorityClass = i;
                maxProbClass = this.columnKappa[i].estimation();
            }
        }
        return majorityClass;
    }

    @Override
    public Measurement[] getPerformanceMeasurements() {
        ArrayList<Measurement> measurements = new ArrayList<Measurement>();
        measurements.add(new Measurement("classified instances", this.getTotalWeightObserved()));
        measurements.add(new Measurement("classifications correct (percent)", this.getFractionCorrectlyClassified() * 100.0));
        measurements.add(new Measurement("Kappa Statistic (percent)", this.getKappaStatistic() * 100.0));
        measurements.add(new Measurement("Kappa Temporal Statistic (percent)", this.getKappaTemporalStatistic() * 100.0));
        measurements.add(new Measurement("Kappa M Statistic (percent)", this.getKappaMStatistic() * 100.0));


        if (precisionRecallOutputOption.isSet())
            measurements.add(new Measurement("F1 Score (percent)", 
                    this.getF1Statistic() * 100.0));
        if (f1PerClassOption.isSet()) {
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("F1 Score for class " + i + 
                        " (percent)", 100.0 * this.getF1Statistic(i)));
            }

            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Kappa Temporal Statistic F1 Score for class " + i +
                        " (percent)", 100.0 * this.getKappaF1TemporalStatistic(i)));
            }
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Kappa M Statistic F1 Score for class " + i +
                        " (percent)", 100.0 * this.getKappaF1MStatistic(i)));
            }
        }
        if (precisionRecallOutputOption.isSet())
            measurements.add(new Measurement("Precision (percent)", 
                this.getPrecisionStatistic() * 100.0));               
        if (precisionPerClassOption.isSet()) {
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Precision for class " + i + 
                        " (percent)", 100.0 * this.getPrecisionStatistic(i)));
            }
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Kappa Precision Temporal Statistic " + i +
                        " (percent)", 100.0 * this.getKappaPrecisionTemporalStatistic(i)));
            }
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Kappa Precision M Statistic " + i +
                        " (percent)", 100.0 * this.getKappaPrecisionMStatistic(i)));
            }
        }
        if (precisionRecallOutputOption.isSet())
            measurements.add(new Measurement("Recall (percent)", 
                this.getRecallStatistic() * 100.0));
        if (recallPerClassOption.isSet()) {
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Recall for class " + i + 
                        " (percent)", 100.0 * this.getRecallStatistic(i)));
            }
            measurements.add(new Measurement("Gmean for recall " +
                    " (percent)", 100.0 * this.getGmeanStatistic()));

            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Kappa Recall Temporal Statistic " + i +
                        " (percent)", 100.0 * this.getKappaRecallTemporalStatistic(i)));
            }
            for (int i = 0; i < this.numClasses; i++) {
                measurements.add(new Measurement("Kappa Recall M Statistic " + i +
                        " (percent)", 100.0 * this.getKappaRecallMStatistic(i)));
            }
            measurements.add(new Measurement("Kappa Gmean Temporal Statistic " +
                    " (percent)", 100.0 * this.getKappaGmeanTemporalStatistic()));
            measurements.add(new Measurement("Kappa Gmean M Statistic " +
                    " (percent)", 100.0 * this.getKappaGmeanMStatistic()));

        }

        Measurement[] result = new Measurement[measurements.size()];

        return measurements.toArray(result);

    }

    public double getTotalWeightObserved() {
        return this.totalWeightObserved;
    }

    public double getFractionCorrectlyClassified() {
        return this.weightCorrect.estimation();
    }

    public double getFractionIncorrectlyClassified() {
        return 1.0 - getFractionCorrectlyClassified();
    }

    public double getKappaStatistic() {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getFractionCorrectlyClassified();
            double pc = 0.0;
            for (int i = 0; i < this.numClasses; i++) {
                pc += this.rowKappa[i].estimation()
                        * this.columnKappa[i].estimation();
            }
            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    public double getKappaTemporalStatistic() {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getFractionCorrectlyClassified();
            double pc = this.weightCorrectNoChangeClassifier.estimation();

            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }
    public double getKappaGmeanTemporalStatistic() {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getGmeanStatistic();
            double pc = this.getGmean_weightCorrectNoChangeClassifierStatistic();

            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    public double getKappaRecallTemporalStatistic(int numClass) {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getRecallStatistic(numClass);
            double pc = this.recall_weightCorrectNoChangeClassifier[numClass].estimation();
            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    public double getKappaPrecisionTemporalStatistic(int numClass) {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getPrecisionStatistic(numClass);
            double pc = this.precision_weightCorrectNoChangeClassifier[numClass].estimation();
            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }
    public double getKappaF1TemporalStatistic(int numClass) {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getF1Statistic(numClass);
            double pc = this.getF1_weightCorrectNoChangeClassifierStatistic(numClass);
            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }
    public double getKappaF1MStatistic(int numClass) {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getF1Statistic(numClass);
            double pc = this.getF1_weightMajorityClassifierStatistic(numClass);
            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    private double getKappaMStatistic() {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getFractionCorrectlyClassified();
            double pc = this.weightMajorityClassifier.estimation();

            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }
    private double getKappaGmeanMStatistic() {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getGmeanStatistic();
            double pc = this.getGmean_weightMajorityClassifierStatistic();

            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    private double getKappaRecallMStatistic(int numClass) {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getRecallStatistic(numClass);
            double pc = this.recall_weightMajorityClassifier[numClass].estimation();

            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }

    private double getKappaPrecisionMStatistic(int numClass) {
        if (this.getTotalWeightObserved() > 0.0) {
            double p0 = getPrecisionStatistic(numClass);
            double pc = this.precision_weightMajorityClassifier[numClass].estimation();

            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }


    public double getPrecisionStatistic() {
        double total = 0;
        for (Estimator ck : this.precision) {
            total += ck.estimation();
        }
        return total / this.precision.length;
    }

    public double getPrecisionStatistic(int numClass) {
        return this.precision[numClass].estimation();
    }

    public double getRecallStatistic() {
        double total = 0;
        for (Estimator ck : this.recall) {
            total += ck.estimation();
        }
        return total / this.recall.length;
    }

    public double getRecallStatistic(int numClass) {
        return this.recall[numClass].estimation();
    }

    public double getPrecision_weightCorrectNoChangeClassifierStatistic() {
        double total = 0;
        for (Estimator ck : this.precision_weightCorrectNoChangeClassifier) {
            total += ck.estimation();
        }
        return total / this.precision_weightCorrectNoChangeClassifier.length;
    }

    public double getPrecision_weightCorrectNoChangeClassifierStatistic(int numClass) {
        return this.precision_weightCorrectNoChangeClassifier[numClass].estimation();
    }

    public double getRecall_weightCorrectNoChangeClassifierStatistic() {
        double total = 0;
        for (Estimator ck : this.recall_weightCorrectNoChangeClassifier) {
            total += ck.estimation();
        }
        return total / this.recall_weightCorrectNoChangeClassifier.length;
    }

    public double getRecall_weightCorrectNoChangeClassifierStatistic(int numClass) {
        return this.recall_weightCorrectNoChangeClassifier[numClass].estimation();
    }


    public double getPrecision_weightMajorityClassifierStatistic() {
        double total = 0;
        for (Estimator ck : this.precision_weightMajorityClassifier) {
            total += ck.estimation();
        }
        return total / this.precision_weightMajorityClassifier.length;
    }

    public double getPrecision_weightMajorityClassifierStatistic(int numClass) {
        return this.precision_weightMajorityClassifier[numClass].estimation();
    }

    public double getRecall_weightMajorityClassifierStatistic() {
        double total = 0;
        for (Estimator ck : this.recall_weightMajorityClassifier) {
            total += ck.estimation();
        }
        return total / this.recall_weightMajorityClassifier.length;
    }

    public double getRecall_weightMajorityClassifierStatistic(int numClass) {
        return this.recall_weightMajorityClassifier[numClass].estimation();
    }

    public double getF1Statistic() {
        return 2 * ((this.getPrecisionStatistic() * this.getRecallStatistic())
                / (this.getPrecisionStatistic() + this.getRecallStatistic()));
    }

    public double getF1_weightMajorityClassifierStatistic(int numClass) {
        return 2 * ((this.getPrecision_weightMajorityClassifierStatistic(numClass) * this.getRecall_weightMajorityClassifierStatistic(numClass))
                / (this.getPrecision_weightMajorityClassifierStatistic(numClass) + this.getRecall_weightMajorityClassifierStatistic(numClass)));
    }

    public double getF1_weightCorrectNoChangeClassifierStatistic(int numClass) {
        return 2 * ((this.getPrecision_weightCorrectNoChangeClassifierStatistic(numClass) * this.getRecall_weightCorrectNoChangeClassifierStatistic(numClass))
                / (this.getPrecision_weightCorrectNoChangeClassifierStatistic(numClass) + this.getRecall_weightCorrectNoChangeClassifierStatistic(numClass)));
    }

    public double getF1Statistic(int numClass) {
        return 2 * ((this.getPrecisionStatistic(numClass) * this.getRecallStatistic(numClass))
                / (this.getPrecisionStatistic(numClass) + this.getRecallStatistic(numClass)));
    }



    public double getGmean_weightMajorityClassifierStatistic(){
        double gmean = 1;
        for (int i = 0; i < this.numClasses; i++) {
            gmean *= this.getRecall_weightMajorityClassifierStatistic(i);
        }
        return Math.sqrt(gmean);
    }
    public double getGmean_weightCorrectNoChangeClassifierStatistic(){
        double gmean = 1;
        for (int i = 0; i < this.numClasses; i++) {
            gmean *= this.getRecall_weightCorrectNoChangeClassifierStatistic(i);
        }
        return Math.sqrt(gmean);
    }
    public double getGmeanStatistic(){
        double gmean = 1;
        for (int i = 0; i < this.numClasses; i++) {
            gmean *= this.getRecallStatistic(i);
        }
        return Math.sqrt(gmean);
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        Measurement.getMeasurementsDescription(getPerformanceMeasurements(),
                sb, indent);
    }

    @Override
    public void addResult(Example<Instance> testInst, Prediction prediction) {
        // TODO Auto-generated method stub

    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

    }

    public interface Estimator extends Serializable {

        void add(double value);

        double estimation();
    }

    public class BasicEstimator implements Estimator {

        protected double len;

        protected double sum;

        @Override
        public void add(double value) {
            if(!Double.isNaN(value)) {
                sum += value;
                len++;
            }
        }

        @Override
        public double estimation() {
            return sum / len;
        }

    }

    protected Estimator newEstimator() {
        return new BasicEstimator();
    }


    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == BasicClassificationPerformanceEvaluator.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
