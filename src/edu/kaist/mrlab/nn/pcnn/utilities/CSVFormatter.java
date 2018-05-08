package edu.kaist.mrlab.nn.pcnn.utilities;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

public class CSVFormatter {

	public void run(String semeval, String ds) throws Exception {
		BufferedWriter bw = Files.newBufferedWriter(Paths.get(ds));
		Reader in = new FileReader(semeval);
		Iterable<CSVRecord> records = CSVFormat.EXCEL.parse(in);
		for (CSVRecord record : records) {
			String sentence = record.get(0);
			String relation = record.get(1);

			int e1SIdx = sentence.indexOf("<e1>");
			int e1EIdx = sentence.indexOf("</e1>");

			int e2SIdx = sentence.indexOf("<e2>");
			int e2EIdx = sentence.indexOf("</e2>");

			String sbj = sentence.substring(e1SIdx + 4, e1EIdx);
			String obj = sentence.substring(e2SIdx + 4, e2EIdx);

			String pred = relation.substring(0, relation.indexOf("("));

			String stc = sentence.replace("<e1>" + sbj + "</e1>", " << _sbj_ >> ");
			stc = stc.replace("<e2>" + obj + "</e2>", " << _obj_ >> ");

			bw.write(sbj + "\t" + obj + "\t" + pred + "\t" + stc + "\n");
		}
	}

	public static void main(String[] ar) throws Exception {
		CSVFormatter csvf = new CSVFormatter();
		csvf.run("data/test/sample.csv", "data/test/sample.ds");
	}
}
