package edu.kaist.mrlab.nn.pcnn.pipeline;

import edu.kaist.mrlab.nn.pcnn.Learner;
import edu.kaist.mrlab.nn.pcnn.prepro.SentencePreprocessorWithTokenizer;
import edu.kaist.mrlab.nn.pcnn.prepro.TrainTestSetSeperator;
import edu.kaist.mrlab.nn.pcnn.utilities.CSVFormatter;
import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;

public class Learning {
	public static void main(String[] ar) throws Exception {
		
		boolean isSemeval = false;

		String root = "data/ds/iter/";
		String semevalFile = root + "semeval-sample";
		String dsInputFilePath = root + "ds-sample";
		final String twitterParsedFolder = root + "total_" + GlobalVariables.option + "/";
		final String trainDevFolder = root + "train_test_" + GlobalVariables.option + "/";
		double devRatio = 0.1;
		
		// Semeval to Ours
		if(isSemeval) {
			CSVFormatter csvf = new CSVFormatter();
			csvf.run(semevalFile, dsInputFilePath);
		}

		// Preparing learning data
		SentencePreprocessorWithTokenizer spwt = new SentencePreprocessorWithTokenizer();
		spwt.run(dsInputFilePath, twitterParsedFolder, true, false);

		TrainTestSetSeperator ttss = new TrainTestSetSeperator(devRatio);
		ttss.run(twitterParsedFolder, trainDevFolder);
		
		// Learning
		Learner learner = new Learner();
		learner.run(trainDevFolder);

	}
}
