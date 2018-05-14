package edu.kaist.mrlab.nn.pcnn;

import java.io.BufferedWriter;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Random;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizerfactory.KoreanTokenizerFactory;
import edu.kaist.mrlab.nn.pcnn.iterator.provider.DataSetIteratorProvider;
import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;
import edu.kaist.mrlab.nn.pcnn.utilities.OntoProperty;
import edu.kaist.mrlab.nn.pcnn.utilities.POSVectors;
import edu.kaist.mrlab.nn.pcnn.utilities.PositionStore;
import edu.kaist.mrlab.nn.pcnn.utilities.PositionVectors;

/**
 * 
 * @author sangha
 *
 */
public class Learner {

	public static PositionStore positionStore = new PositionStore();
	public static final String WORD_VECTORS_PATH = "data/embedding/ko_vec_100dim_1min_title_" + GlobalVariables.option;
	// public static final String WORD_VECTORS_PATH =
	// "data/embedding/ko_vec_100dim_1min_word_stem";

	public void run(String trainDevFolder) throws Exception {
		
		DataSetIteratorProvider dsip = new DataSetIteratorProvider();

		// Basic configuration
		int batchSize = GlobalVariables.batchSize;
		int vectorSize = GlobalVariables.vectorSize; // Size of the word vectors
		int nEpochs = GlobalVariables.nEpochs; // Number of epochs (full passes of training data) to train on
		int truncateReviewsToLength = GlobalVariables.truncateSentencesToLength; // Truncate reviews with length (#
																					// words) greater than this
		int numOfRelation = OntoProperty.values().length;

		int cnnLayerFeatureMaps = GlobalVariables.cnnLayerFeatureMaps; // Number of feature maps / channels / depth for
																		// each CNN layer
		PoolingType poolingType = GlobalVariables.poolingType;
		Random rng = GlobalVariables.rng; // For shuffling repeatability

		boolean saveUpdater = true;

		ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder().weightInit(WeightInit.RELU)
				.activation(Activation.LEAKYRELU) // non-linear function (ex, Affine --> RELU --> Affine --> RELU)
				.updater(Updater.ADAM).convolutionMode(ConvolutionMode.Same).regularization(true).l2(0.0001) // h in
																												// numerical
																												// diff.
																												// function,
																												// f(x+h)
																												// -
																												// f(x-h)
																												// / 2*h
				.learningRate(0.01) // eta in gradient descent method
				.graphBuilder().addInputs("input")
				.addLayer("cnn1",
						new ConvolutionLayer.Builder().kernelSize(3, vectorSize).stride(1, vectorSize).nIn(1)
								.nOut(cnnLayerFeatureMaps).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder().kernelSize(3, vectorSize).stride(1, vectorSize).nIn(1)
								.nOut(cnnLayerFeatureMaps).build(),
						"input")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder().kernelSize(3, vectorSize).stride(1, vectorSize).nIn(1)
								.nOut(cnnLayerFeatureMaps).build(),
						"input")
				.addVertex("merge", new MergeVertex(), "cnn1", "cnn2", "cnn3") // concatenation
				.addLayer("pooling", new GlobalPoolingLayer.Builder().poolingType(poolingType).build(), "merge")
				.addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
						.activation(Activation.SOFTMAX).nIn(3 * cnnLayerFeatureMaps).nOut(numOfRelation).build(),
						"pooling")
				.setOutputs("out").build();

		ComputationGraph net = new ComputationGraph(config);
		net.init();

		System.out.println("Number of parameters by layer:");
		for (Layer l : net.getLayers()) {
			System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
		}

		System.out.println("Loading word and position vectors, and then creating DataSetIterators");
		TokenizerFactory t = new KoreanTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		PositionVectors positionVectors = new PositionVectors();
		POSVectors POSVectors = new POSVectors();

		long start = System.currentTimeMillis();
		WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(WORD_VECTORS_PATH);

		DataSetIterator trainIter = dsip.getDataSetIterator(true, wordVectors, positionVectors, POSVectors, batchSize,
				truncateReviewsToLength, rng, t, trainDevFolder);
		DataSetIterator testIter = dsip.getDataSetIterator(false, wordVectors, positionVectors, POSVectors, batchSize,
				truncateReviewsToLength, rng, t, trainDevFolder);
		long end = System.currentTimeMillis();

		System.out.println(
				"Embedding vectors and T/T data set loading completed. Takes " + (end - start) / 1000 + "sec.");

		BufferedWriter bw = Files.newBufferedWriter(Paths.get("data/log/trainAcc.log"), StandardOpenOption.CREATE);

		System.out.println("Starting training");
		for (int i = 0; i < nEpochs; i++) {
			start = System.currentTimeMillis();
			net.fit(trainIter);
			end = System.currentTimeMillis();
			System.out.println("Epoch " + i + " complete. Takes " + (end - start) / 1000 + "sec.");
			System.out.println("Starting evaluation:");
			Learner.positionStore.clear();
			start = System.currentTimeMillis();
			Evaluation evaluation = net.evaluate(testIter);
			end = System.currentTimeMillis();

			System.out.println("Evaluation completed. Takes " + (end - start) / 1000 + "sec.");
			System.out.println(evaluation.stats());
			bw.write(evaluation.stats());
		}

		System.out.println("Saving model");
		// to save the model
		File locationToSave = new File("data/model/CNN_RELU_LEAKYRELU_ADAM_" + GlobalVariables.option + ".zip");

		ModelSerializer.writeModel(net, locationToSave, saveUpdater);

		System.out.println("Complete.");
		bw.close();
	}

	public static void main(String[] ar) throws Exception {

		String dsRoot = "data/ds/iter/";
		String trainDevFolder = dsRoot + "train_test_" + GlobalVariables.option + "/";

		Learner learner = new Learner();
		learner.run(trainDevFolder);
	}
}
