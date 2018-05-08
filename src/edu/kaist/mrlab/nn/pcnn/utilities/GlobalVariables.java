package edu.kaist.mrlab.nn.pcnn.utilities;

import java.util.Random;

import org.deeplearning4j.nn.conf.layers.PoolingType;

public class GlobalVariables {
	
	// process option
	public static boolean isTraining = true;
	
	// model learning
	public static int batchSize = 16;
	public static int vectorSize = 110; // Size of the word vectors
	public static int nEpochs = 1; // Number of epochs (full passes of training data) to train on
	public static int truncateSentencesToLength = 80; // Truncate reviews with length (# words) greater than this
	
	public static int cnnLayerFeatureMaps = 230; // Number of feature maps / channels / depth for each CNN layer
	public static PoolingType poolingType = PoolingType.MAX;
	public static Random rng = new Random(123); // For shuffling repeatability
	
	// which data
	public static String option = "pos_stem";
	
}
