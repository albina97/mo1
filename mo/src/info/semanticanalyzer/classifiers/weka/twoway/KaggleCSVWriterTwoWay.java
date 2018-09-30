package info.semanticanalyzer.classifiers.weka.twoway;

import info.semanticanalyzer.classifiers.weka.SentimentClass;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class KaggleCSVWriterTwoWay {
    public static final String CSV_HEADER = "PhraseId,Sentiment";
    BufferedWriter bw;

    public KaggleCSVWriterTwoWay(String csvFile) throws IOException {
        bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(csvFile), "utf8"));
        bw.write(CSV_HEADER);
        bw.write("\n");
    }

    public void writeKaggleCSV(KaggleCSVReaderTwoWay.CSVInstanceTwoWay csvInstanceThreeWay) throws IOException {
        try {
            bw.write(String.valueOf(csvInstanceThreeWay.phraseID));
            bw.write(",");
            bw.write(String.valueOf(sentiment.ordinal()));
            bw.write("\n");
        } catch (IOException e) {
            close();
            throw e;
        }
    }

    public void close() throws IOException {
        bw.close();
    }
}
