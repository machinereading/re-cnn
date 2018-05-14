package edu.kaist.mrlab.nn.pcnn.pipeline;

import edu.kaist.mrlab.nn.pcnn.Configuration;
import edu.kaist.mrlab.nn.pcnn.Testor;
import edu.kaist.mrlab.nn.pcnn.postpro.CSVWriter;
import edu.kaist.mrlab.nn.pcnn.prepro.SentencePreprocessorWithTokenizer;
import edu.kaist.mrlab.nn.pcnn.prepro.TrainTestSetSeperator;
import edu.kaist.mrlab.nn.pcnn.utilities.CSVFormatter;

public class Extraction {
	public static void test() throws Exception {

		boolean isSemeval = Configuration.isSemeval;

		String semevalTestFile = Configuration.semevalTestFile;
		String testInputFilePath = Configuration.testInputFilePath;
		final String twitterParsedFolder = Configuration.twitterParsedFolder;
		final String testFolder = Configuration.testFolder;
		double testRatio = Configuration.testRatio;
		boolean isLive = Configuration.isLive;

		// Semeval to Ours
		if (isSemeval) {
			CSVFormatter csvf = new CSVFormatter();
			csvf.run(semevalTestFile, testInputFilePath);
		}
		
		if(!isLive) {
			// Preparing learning data
			SentencePreprocessorWithTokenizer spwt = new SentencePreprocessorWithTokenizer();
			spwt.run(testInputFilePath, twitterParsedFolder, true, false);

			TrainTestSetSeperator ttss = new TrainTestSetSeperator(testRatio);
			ttss.run(twitterParsedFolder, testFolder);
		}
		
		// Extract triples
		Testor t = new Testor();
		t.init();
		t.run(isLive, testInputFilePath, testFolder);
		
		if(isSemeval) {
			CSVWriter csvw = new CSVWriter();
			csvw.run();
		}

	}
}
