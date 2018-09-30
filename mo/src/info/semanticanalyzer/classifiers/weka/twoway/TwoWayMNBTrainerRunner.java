package info.semanticanalyzer.classifiers.weka.twoway;

import info.semanticanalyzer.classifiers.weka.SentimentClass;

public class TwoWayMNBTrainerRunner {
    public static void main(String[] args) throws Exception {
        KaggleCSVReaderTwoWay kaggleCSVReaderTwoWay = new KaggleCSVReaderTwoWay();
        kaggleCSVReaderTwoWay.readKaggleCSV("kaggle/train.tsv");
        KaggleCSVReaderTwoWay.CSVInstanceTwoWay csvInstanceTwoWay;

        String outputModel = "models/two-way-sentiment-mnb.model";

        TwoClassTrainer twoWayMNBTrainer = new TwoClassTrainer(outputModel);

        int sentimentPositiveCount = 0;
        int sentimentNegativeCount = 0;

        System.out.println("Adding training instances");
        int addedNum = 0;
        while ((csvInstanceTwoWay = kaggleCSVReaderTwoWay.next()) != null) {
            if (csvInstanceTwoWay.isValidInstance) {
                if (csvInstanceTwoWay.sentiment.equals(SentimentClass.TwoWayClazz.SWINDLER) && sentimentPositiveCount < 7072) {
                    sentimentPositiveCount++;
                    twoWayMNBTrainer.addTrainingInstance(csvInstanceTwoWay.sentiment, csvInstanceTwoWay.phrase.split("\\s+"));
                    addedNum++;
                }
                else if (csvInstanceTwoWay.sentiment.equals(SentimentClass.TwoWayClazz.AVERAGE) && sentimentNegativeCount < 7072) {
                    sentimentNegativeCount++;
                    twoWayMNBTrainer.addTrainingInstance(csvInstanceTwoWay.sentiment, csvInstanceTwoWay.phrase.split("\\s+"));
                    addedNum++;
                }

                if (sentimentPositiveCount >= 7072 && sentimentNegativeCount >= 7072)
                    break;
            }
        }

        kaggleCSVReaderTwoWay.close();

        System.out.println("Added " + addedNum + " instances");
        System.out.println("Of which " + sentimentPositiveCount + " positive instances, " +
                sentimentNegativeCount + " negative instances and ");

        System.out.println("Training and saving Model");
        twoWayMNBTrainer.trainModel();
        twoWayMNBTrainer.saveModel();

        System.out.println("Testing model");
        twoWayMNBTrainer.testModel();
    }

}
