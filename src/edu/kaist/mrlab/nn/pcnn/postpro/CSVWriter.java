package edu.kaist.mrlab.nn.pcnn.postpro;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.StringTokenizer;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

public class CSVWriter {
	public void run() throws Exception {

		BufferedReader br = Files.newBufferedReader(Paths.get("data/test/pl4-out"));
		BufferedWriter bw = Files.newBufferedWriter(Paths.get("data/test/pl4-out.csv"));
		CSVPrinter csvPrinter = new CSVPrinter(bw, CSVFormat.DEFAULT);
		String input = null;
		while ((input = br.readLine()) != null) {

			StringTokenizer st = new StringTokenizer(input, "\t");
			String sbj = st.nextToken();
			String pred = st.nextToken();
			String obj = st.nextToken();
			st.nextToken(); // dot
			String scr = st.nextToken();
			String stc = st.nextToken();

			String semStc = stc.replace(" << " + sbj + " >> ", "<e1>" + sbj + "</e1>");
			semStc = semStc.replace(" << " + obj + " >> ", "<e2>" + obj + "</e2>");
			
			String semRel = pred + "(e1,e2)";

			csvPrinter.printRecord(semStc, semRel, scr);

		}

		br.close();
		csvPrinter.close();
		bw.close();

	}

	public static void main(String[] ar) throws Exception {
		CSVWriter csvw = new CSVWriter();
		csvw.run();
	}
}
