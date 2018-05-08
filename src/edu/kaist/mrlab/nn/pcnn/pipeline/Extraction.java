package edu.kaist.mrlab.nn.pcnn.pipeline;

import edu.kaist.mrlab.nn.pcnn.Testor;
import edu.kaist.mrlab.nn.pcnn.prepro.SentencePreprocessorWithTokenizer;
import edu.kaist.mrlab.nn.pcnn.prepro.TrainTestSetSeperator;
import edu.kaist.mrlab.nn.pcnn.utilities.CSVFormatter;
import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;

public class Extraction {
	public static void main(String[] ar) throws Exception {

		boolean isSemeval = false;

		String root = "data/ds/iter/";
		String semevalTestFile = root + "semeval-test";
//		String testInputFilePath = root + "gs-sample";
		String testInputFilePath = "data/test/ko_test.txt"; // if live, sample file
		final String twitterParsedFolder = root + "total_" + GlobalVariables.option + "/";
		final String testFolder = root + "train_test_" + GlobalVariables.option + "/";
		double testRatio = 1.0;
		boolean isLive = true;

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

	}
}
