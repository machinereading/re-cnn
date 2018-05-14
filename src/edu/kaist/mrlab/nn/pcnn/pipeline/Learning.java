package edu.kaist.mrlab.nn.pcnn.pipeline;

import edu.kaist.mrlab.nn.pcnn.Configuration;
import edu.kaist.mrlab.nn.pcnn.Learner;
import edu.kaist.mrlab.nn.pcnn.prepro.SentencePreprocessorWithTokenizer;
import edu.kaist.mrlab.nn.pcnn.prepro.TrainTestSetSeperator;
import edu.kaist.mrlab.nn.pcnn.utilities.CSVFormatter;

public class Learning {
	public static void learn() throws Exception {

		boolean isSemeval = Configuration.isSemeval;

		String semevalFile = Configuration.semevalFile;
		String dsInputFilePath = Configuration.dsInputFilePath;
		final String twitterParsedFolder = Configuration.twitterParsedFolder;
		final String trainDevFolder = Configuration.trainDevFolder;
		double devRatio = Configuration.devRatio;

		// Semeval to Ours
		if (isSemeval) {
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
