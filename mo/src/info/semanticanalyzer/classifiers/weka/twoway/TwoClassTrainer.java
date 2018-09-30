package info.semanticanalyzer.classifiers.weka.twoway;

import com.google.inject.internal.util.Join;
import info.semanticanalyzer.classifiers.weka.SentimentClass;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

public class TwoClassTrainer {
    private NaiveBayesMultinomialText classifier;
    private String modelFile;
    private Instances dataRaw;

    public TwoClassTrainer(String outputModel) {
        classifier = new NaiveBayesMultinomialText();
        modelFile = outputModel;

        ArrayList<Attribute> atts = new ArrayList<Attribute>(2);
        ArrayList<String> classVal = new ArrayList<String>();
        classVal.add(SentimentClass.TwoWayClazz.SWINDLER.name());
        classVal.add(SentimentClass.TwoWayClazz.AVERAGE.name());
        atts.add(new Attribute("content",(ArrayList<String>)null));
        atts.add(new Attribute("@@class@@",classVal));

        dataRaw = new Instances("TrainingInstances",atts,10);
    }

    public void addTrainingInstance(SentimentClass.TwoWayClazz twoWayClazz, String[] words) {
        double[] instanceValue = new double[dataRaw.numAttributes()];
        instanceValue[0] = dataRaw.attribute(0).addStringValue(Join.join(" ", words));
        instanceValue[1] = twoWayClazz.ordinal();
        dataRaw.add(new DenseInstance(1.0, instanceValue));
        dataRaw.setClassIndex(1);
    }

    public void trainModel() throws Exception {
        classifier.buildClassifier(dataRaw);
    }

    public void testModel() throws Exception {
        Evaluation eTest = new Evaluation(dataRaw);
        eTest.evaluateModel(classifier, dataRaw);
        String strSummary = eTest.toSummaryString();
        System.out.println(strSummary);
    }

    public void showInstances() {
        System.out.println(dataRaw);
    }

    public Instances getDataRaw() {
        return dataRaw;
    }

    public void saveModel() throws Exception {
        weka.core.SerializationHelper.write(modelFile, classifier);
    }

    public void loadModel(String _modelFile) throws Exception {
        NaiveBayesMultinomialText classifier = (NaiveBayesMultinomialText) weka.core.SerializationHelper.read(_modelFile);
        this.classifier = classifier;
    }

    public SentimentClass.TwoWayClazz classify(String sentence) throws Exception {
        double[] instanceValue = new double[dataRaw.numAttributes()];
        instanceValue[0] = dataRaw.attribute(0).addStringValue(sentence);

        Instance toClassify = new DenseInstance(1.0, instanceValue);
        dataRaw.setClassIndex(1);
        toClassify.setDataset(dataRaw);

        double prediction = this.classifier.classifyInstance(toClassify);

        double distribution[] = this.classifier.distributionForInstance(toClassify);

        return SentimentClass.TwoWayClazz.values()[(int)prediction];
    }

}
