package info.semanticanalyzer.classifiers.weka.twoway;


import info.semanticanalyzer.classifiers.weka.SentimentClass;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class KaggleCSVReaderTwoWay {
    private String line;
    private CSVInstanceTwoWay csvInstanceTwoWay;
    private int step = 0;

    private BufferedReader br;

    private int showStatsAt = 1000;

    void readKaggleCSV(String csvFile) throws IOException {
        br = new BufferedReader(new FileReader(csvFile));

        line = br.readLine();

        if (line != null) {
            if (line.startsWith("PhraseId")) {
                line = br.readLine();
            }

            if (line != null) {
                extractInstance();
            }
        }
    }

    private void extractInstance() {
        String[] attrs = line.split("\t");

        if (csvInstanceTwoWay == null) {
            csvInstanceTwoWay = new CSVInstanceTwoWay();
        }
        csvInstanceTwoWay.phraseID = Integer.valueOf(attrs[0]);
        csvInstanceTwoWay.sentenceID = Integer.valueOf(attrs[1]);
        csvInstanceTwoWay.phrase = attrs[2];
        // there is additionally sentiment tag for training data
        if (attrs.length > 3) {
            Integer sentimentOrdinal = Integer.valueOf(attrs[3]);

            if (sentimentOrdinal <= 1) {
                csvInstanceTwoWay.sentiment = SentimentClass.TwoWayClazz.values()[sentimentOrdinal];
                csvInstanceTwoWay.isValidInstance = true;
            } else {
                // can't process the instance, because the sentiment ordinal is out of the acceptable range of two classes
                csvInstanceTwoWay.isValidInstance = false;
            }
        }
    }

    CSVInstanceTwoWay next() {
        if (step == 0) {
            step++;
            return csvInstanceTwoWay;
        }

        if (step % showStatsAt == 0) {
            System.out.println("Processed instances: " + step);
        }

        try {
            line = br.readLine();
            if (line != null) {
                extractInstance();
            } else {
                return null;
            }
            step++;
            return csvInstanceTwoWay;
        } catch (IOException e) {
            return null;
        }
    }

    void close() {
        try {
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    class CSVInstanceTwoWay {
        int phraseID;
        int sentenceID;
        String phrase;
        SentimentClass.TwoWayClazz sentiment;
        boolean isValidInstance;

        @Override
        public String toString() {
            return "CSVInstanceTwoWay{" +
                    "phraseID=" + phraseID +
                    ", sentenceID=" + sentenceID +
                    ", phrase='" + phrase + '\'' +
                    ", sentiment=" + sentiment +
                    '}';
        }
    }

}



