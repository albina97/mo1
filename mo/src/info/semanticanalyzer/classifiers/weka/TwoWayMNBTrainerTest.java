package info.semanticanalyzer.classifiers.weka;

import info.semanticanalyzer.classifiers.weka.twoway.TwoClassTrainer;
import junit.framework.Assert;
import org.apache.commons.io.IOUtils;
import org.junit.Test;

import java.io.FileInputStream;

public class TwoWayMNBTrainerTest {
    TwoClassTrainer twoClassTrainer;
    String modelFile = "models/two-way-sentiment-mnb.model";
    private static final String PERFOMRANCE_TEST_CONTENT_FILE = "src/test/resources/creditcard.csv";

    @org.junit.Before
    public void setUp() throws Exception {
        twoClassTrainer = new TwoClassTrainer(modelFile);
    }

    @org.junit.Test
    public void testAddTrainingInstance() throws Exception {
        twoClassTrainer.addTrainingInstance(SentimentClass.TwoWayClazz.AVERAGE, new String[] {"average"});
        twoClassTrainer.addTrainingInstance(SentimentClass.TwoWayClazz.SWINDLER, new String[] {"swindler"});
        twoClassTrainer.showInstances();
    }

    @org.junit.Test
    public void testTrainModel() throws Exception {
        twoClassTrainer.addTrainingInstance(SentimentClass.TwoWayClazz.AVERAGE, new String[] {"average"});
        twoClassTrainer.addTrainingInstance(SentimentClass.TwoWayClazz.SWINDLER, new String[] {"swindler"});
        twoClassTrainer.trainModel();
        twoClassTrainer.testModel();
    }

    @org.junit.Test
    public void testSaveModel() throws Exception {
        twoClassTrainer.addTrainingInstance(SentimentClass.TwoWayClazz.AVERAGE, new String[] {"average"});
        twoClassTrainer.addTrainingInstance(SentimentClass.TwoWayClazz.SWINDLER, new String[] {"swindler"});
        twoClassTrainer.trainModel();
        twoClassTrainer.testModel();
        twoClassTrainer.saveModel();
        System.out.println("===== Loading and testing model ====");
        twoClassTrainer.loadModel(modelFile);
        twoClassTrainer.testModel();
    }

    @org.junit.Test
    public void testExistingModel() throws Exception {
        twoClassTrainer.addTrainingInstance(SentimentClass.TwoWayClazz.AVERAGE, new String[] {"average"});
        twoClassTrainer.addTrainingInstance(SentimentClass.TwoWayClazz.SWINDLER, new String[] {"swindler"});
        twoClassTrainer.loadModel(modelFile);
        twoClassTrainer.testModel();
    }

    @org.junit.Test
    public void testArbitraryTextPositive() throws Exception {
        twoClassTrainer.loadModel(modelFile);
        Assert.assertEquals(SentimentClass.TwoWayClazz.AVERAGE, twoClassTrainer.classify("0"));
    }

    @org.junit.Test
    public void testArbitraryTextNegative() throws Exception {
        twoClassTrainer.loadModel(modelFile);
        Assert.assertEquals(SentimentClass.TwoWayClazz.SWINDLER, twoClassTrainer.classify("1"));
    }

    @Test
    public void testPerformance() throws Exception
    {
        String content = IOUtils.toString(new FileInputStream(PERFOMRANCE_TEST_CONTENT_FILE), "UTF-8");
        String[] lines = content.split("\n");

        int wordsCount = getWordsCount(lines);

        twoClassTrainer.loadModel(modelFile);

        test(lines, wordsCount, content.length()); // warm up

        test(lines, wordsCount, content.length()); // test
    }

    private int getWordsCount(String[] texts)
    {
        int count = 0;
        for (String str : texts) {
            count += str.split("\\s+").length;
        }
        return count;
    }

    private void test(String[] texts, int wordsCount, int totalLength) throws Exception {
        System.out.println("Testing on " + texts.length + " samples, " + wordsCount + " words, " + totalLength
                + " characters...");

        long startTime = System.currentTimeMillis();
        for (String str : texts) {
            // to print out the predicted labels, uncomment the line:
            //System.out.println(threeWayMnbTrainer.classify(str).name());
            twoClassTrainer.classify(str).name();
        }
        long elapsedTime = System.currentTimeMillis() - startTime;

        System.out.println("Time " + elapsedTime + " ms.");
        System.out.println("Speed " + ((double) totalLength / elapsedTime) + " chars/ms");
        System.out.println("Speed " + ((double) wordsCount / elapsedTime) + " words/ms");
        System.out.println("+++++++++=");
    }
}

