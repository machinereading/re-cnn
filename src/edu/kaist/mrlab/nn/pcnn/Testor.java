package edu.kaist.mrlab.nn.pcnn;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Map;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizerfactory.KoreanTokenizerFactory;
import edu.kaist.mrlab.nn.pcnn.eval.ComputationGraphEvaluator;
import edu.kaist.mrlab.nn.pcnn.iterator.provider.DataSetIteratorProvider;
import edu.kaist.mrlab.nn.pcnn.postpro.DomainRangeChecker;
import edu.kaist.mrlab.nn.pcnn.prepro.ExtractorPreprocessor;
import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;
import edu.kaist.mrlab.nn.pcnn.utilities.POSVectors;
import edu.kaist.mrlab.nn.pcnn.utilities.PositionStore;
import edu.kaist.mrlab.nn.pcnn.utilities.PositionVectors;
import edu.kaist.mrlab.nn.pcnn.utilities.SentenceStore;

public class Testor {

	public static final String WORD_VECTORS_PATH = "data/embedding/ko_vec_100dim_1min_title_" + GlobalVariables.option;
	public static SentenceStore sentenceStore;
	public static PositionStore positionStore;
	public static ExtractorPreprocessor ep;
	public static DataSetIteratorProvider dsip;
	public static TokenizerFactory t;
	public static PositionVectors positionVectors;
	public static POSVectors POSVectors;
	public static WordVectors wordVectors;
	public static File locationToLoad;
	public static ComputationGraph net;
	public static DomainRangeChecker drc;
	public static Map<String, Integer> labelClassMap;

	public static SentenceStore originalSentenceStore;

	public void init() throws Exception {

		GlobalVariables.isTraining = false;

		ep = new ExtractorPreprocessor();
		dsip = new DataSetIteratorProvider();
		t = new KoreanTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		positionVectors = new PositionVectors();
		POSVectors = new POSVectors();
		wordVectors = WordVectorSerializer.readWord2VecModel(WORD_VECTORS_PATH);

		locationToLoad = new File("data/model/CNN_RELU_LEAKYRELU_ADAM_" + GlobalVariables.option + ".zip");
		net = ModelSerializer.restoreComputationGraph(locationToLoad);

		drc = new DomainRangeChecker();
		drc.loadEntityType();
		drc.loadPropertyDomainRange();
		drc.loadEntityCount();

	}

	public void run(boolean isLive, String extract, String root) throws Exception {

		originalSentenceStore = new SentenceStore();
		sentenceStore = new SentenceStore();
		positionStore = new PositionStore();

		if (isLive) {
			ep.test(extract);

			DataSetIterator testIter = dsip.getTestDataSetIterator(false, wordVectors, positionVectors, POSVectors,
					GlobalVariables.batchSize, GlobalVariables.truncateSentencesToLength, GlobalVariables.rng, t);
			ComputationGraphEvaluator evaluator = new ComputationGraphEvaluator(net);
			evaluator.evaluate(testIter);
			drc.filter();

		} else {
			DataSetIterator testIter = dsip.getDataSetIterator(false, wordVectors, positionVectors, POSVectors,
					GlobalVariables.batchSize, GlobalVariables.truncateSentencesToLength, GlobalVariables.rng, t, root);
			Evaluation evaluation = net.evaluate(testIter);
			System.out.println(evaluation.stats());

			testIter = dsip.getDataSetIterator(false, wordVectors, positionVectors, POSVectors,
					GlobalVariables.batchSize, GlobalVariables.truncateSentencesToLength, GlobalVariables.rng, t, root);
			ComputationGraphEvaluator evaluator = new ComputationGraphEvaluator(net);
			evaluator.evaluate(testIter);

			fileCopy("data/test/pl4-out-orig", "data/test/pl4-out");
		}

	}

	public static void fileCopy(String inFileName, String outFileName) {
		try {
			FileInputStream fis = new FileInputStream(inFileName);
			FileOutputStream fos = new FileOutputStream(outFileName);

			int data = 0;
			while ((data = fis.read()) != -1) {
				fos.write(data);
			}
			fis.close();
			fos.close();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void main(String[] ar) throws Exception {

		Testor t = new Testor();
		t.init();
		t.run(false, "data/test/ko_test_parsed.txt", "");

	}
}
