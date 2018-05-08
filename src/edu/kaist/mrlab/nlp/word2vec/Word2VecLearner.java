package edu.kaist.mrlab.nlp.word2vec;

import java.io.File;
import java.nio.file.Paths;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizerfactory.KoreanTokenizerFactory;
import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;

/**
 *
 * Neural net that processes text into wordvectors. See below url for an
 * in-depth explanation. https://deeplearning4j.org/word2vec.html
 * 
 */
public class Word2VecLearner {

	private static Logger log = LoggerFactory.getLogger(Word2VecLearner.class);
	
	public static void main(String[] args) throws Exception {
		// Gets Path to Text file
		String filePath = Paths.get("data/corpus/olympic-embedding-corpus.txt").toString();
		
		log.info("Load & Vectorize Sentences....");
		// Strip white space before and after for each line
		SentenceIterator iter = new BasicLineIterator(filePath);
		// Split on white spaces in the line to get words
		TokenizerFactory t = new KoreanTokenizerFactory();

		/*
		 * CommonPreprocessor will apply the following regex to each token:
		 * [\d\.:,"'\(\)\[\]|/?!;]+ So, effectively all numbers, punctuation
		 * symbols and some special symbols are stripped off. Additionally it
		 * forces lower case for all tokens.
		 */
		t.setTokenPreProcessor(new CommonPreprocessor());

		log.info("Building model....");
		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).iterations(1).layerSize(100).seed(42).windowSize(5)
				.iterate(iter).tokenizerFactory(t).build();

		log.info("Fitting Word2Vec model....");
		vec.fit();

		log.info("Writing word vectors to text file....");

		// Write word vectors to file
		WordVectorSerializer.writeWord2VecModel(vec, new File("data/embedding/ko_vec_100dim_1min_olympic_" + GlobalVariables.option));

		// Prints out the closest 10 words to "day". An example on what to do
		// with these Word Vectors.
		Collection<String> lst = vec.wordsNearest("임효준/Entity", 10);
		System.out.println("10 Words closest to '임효준/Entity': " + lst);
		
		lst = vec.wordsNearest("평창/Entity", 10);
		System.out.println("10 Words closest to '평창/Entity': " + lst);

		lst = vec.wordsNearest("강원도/Entity", 10);
		System.out.println("10 Words closest to '강원도/Entity': " + lst);

		lst = vec.wordsNearest("올림픽/Entity", 10);
		System.out.println("10 Words closest to '올림픽/Entity': " + lst);

		// TODO resolve missing UiServer
		// UiServer server = UiServer.getInstance();
		// System.out.println("Started on port " + server.getPort());
	}
}
