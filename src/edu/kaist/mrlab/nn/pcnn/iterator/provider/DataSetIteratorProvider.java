package edu.kaist.mrlab.nn.pcnn.iterator.provider;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import edu.kaist.mrlab.nn.pcnn.Testor;
import edu.kaist.mrlab.nn.pcnn.iterator.LabeledSentencePositionProvider;
import edu.kaist.mrlab.nn.pcnn.iterator.SentencePositionDataSetIterator;
import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;
import edu.kaist.mrlab.nn.pcnn.utilities.OntoProperty;
import edu.kaist.mrlab.nn.pcnn.utilities.POSVectors;
import edu.kaist.mrlab.nn.pcnn.utilities.PositionVectors;

public class DataSetIteratorProvider {

	public DataSetIterator getDataSetIterator(boolean isTraining, WordVectors wordVectors,
			PositionVectors positionVectors, POSVectors POSVectors, int minibatchSize, int maxSentenceLength, Random rng,
			TokenizerFactory tokenizerFactory, String trainDevFolder) throws Exception {

		List<String> sentences = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		List<Integer> sbjPosList = new ArrayList<Integer>();
		List<Integer> objPosList = new ArrayList<Integer>();

		for (OntoProperty p : OntoProperty.values()) {
			String property = p.toString();

			if (property.equals("classP")) {
				property = "class";
			}

			Path path = null;

			if (isTraining) {
				path = Paths.get(trainDevFolder, property + "_train");
			} else {
				path = Paths.get(trainDevFolder, property + "_test");
			}

			BufferedReader br = Files.newBufferedReader(path);
			String input = null;
			while ((input = br.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(input, "\t");
				String stc = st.nextToken();
				int sbjPos = Integer.parseInt(st.nextToken());
				int objPos = Integer.parseInt(st.nextToken());
				String label = st.nextToken();
				String origStc = st.nextToken();

				String[] stcArr = stc.split(" ");
				if (stcArr.length > GlobalVariables.truncateSentencesToLength - 1) {
					continue;
				}

				if (sbjPos == objPos) {
					continue;
				}

				sentences.add(stc);
				labels.add(label);
				sbjPosList.add(sbjPos);
				objPosList.add(objPos);
				
				Testor.originalSentenceStore.addSentence(origStc);

			}

			br.close();
		}

		LabeledSentencePositionProvider sentenceProvider = new CollectionLabeledSentencePositionProvider(sentences,
				sbjPosList, objPosList, labels);

		return new SentencePositionDataSetIterator.Builder().sentenceProvider(sentenceProvider).wordVectors(wordVectors)
				.positionVectors(positionVectors).POSVectors(POSVectors).minibatchSize(minibatchSize).maxSentenceLength(maxSentenceLength)
				.useNormalizedWordVectors(false).build();
	}

	public DataSetIterator getTestDataSetIterator(boolean isTraining, WordVectors wordVectors,
			PositionVectors positionVectors, POSVectors POSVectors, int minibatchSize, int maxSentenceLength, Random rng,
			TokenizerFactory tokenizerFactory) throws Exception {

		List<String> sentences = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		List<Integer> sbjPosList = new ArrayList<Integer>();
		List<Integer> objPosList = new ArrayList<Integer>();

		String path = "data/test/ko_test_parsed.txt";

		BufferedReader br = Files.newBufferedReader(Paths.get(path));
		String input = null;
		while ((input = br.readLine()) != null) {

			StringTokenizer st = new StringTokenizer(input, "\t");
			String stc = st.nextToken();
			int sbjPos = Integer.parseInt(st.nextToken());
			int objPos = Integer.parseInt(st.nextToken());
			String label = st.nextToken();
			String origStc = st.nextToken();

			String[] stcArr = stc.split(" ");
			if (stcArr.length > GlobalVariables.truncateSentencesToLength - 1) {
				continue;
			}

			if (sbjPos == objPos) {
				continue;
			}

			sentences.add(stc);
			labels.add(label);
			sbjPosList.add(sbjPos);
			objPosList.add(objPos);
			Testor.originalSentenceStore.addSentence(origStc);

		}

		br.close();

		LabeledSentencePositionProvider sentenceProvider = new CollectionLabeledSentencePositionProvider(sentences,
				sbjPosList, objPosList, labels);

		return new SentencePositionDataSetIterator.Builder().sentenceProvider(sentenceProvider).wordVectors(wordVectors)
				.positionVectors(positionVectors).minibatchSize(minibatchSize).maxSentenceLength(maxSentenceLength)
				.useNormalizedWordVectors(false).build();

	}
	
	public DataSetIterator getGSDataSetIterator(boolean isTraining, WordVectors wordVectors,
			PositionVectors positionVectors, POSVectors POSVectors, int minibatchSize, int maxSentenceLength, Random rng,
			TokenizerFactory tokenizerFactory) throws Exception {

		List<String> sentences = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		List<Integer> sbjPosList = new ArrayList<Integer>();
		List<Integer> objPosList = new ArrayList<Integer>();

		for (OntoProperty p : OntoProperty.values()) {
			String property = p.toString();

			if (property.equals("classP")) {
				property = "class";
			}

			String path = null;

			if (isTraining) {
				path = "data/gs/kbox-wiki/train_test_" + GlobalVariables.option + "/" + property + "_train";
			} else {
				path = "data/gs/kbox-wiki/train_test_" + GlobalVariables.option + "/" + property + "_test";
			}

			BufferedReader br = Files.newBufferedReader(Paths.get(path));
			String input = null;
			while ((input = br.readLine()) != null) {

				StringTokenizer st = new StringTokenizer(input, "\t");
				String stc = st.nextToken();
				int sbjPos = Integer.parseInt(st.nextToken());
				int objPos = Integer.parseInt(st.nextToken());
				String label = st.nextToken();
				String origStc = st.nextToken();

				String[] stcArr = stc.split(" ");
				if (stcArr.length > GlobalVariables.truncateSentencesToLength - 1) {
					continue;
				}

				if (sbjPos == objPos) {
					continue;
				}

				sentences.add(stc);
				labels.add(label);
				sbjPosList.add(sbjPos);
				objPosList.add(objPos);
				Testor.originalSentenceStore.addSentence(origStc);
			}

			br.close();
		}

		LabeledSentencePositionProvider sentenceProvider = new CollectionLabeledSentencePositionProvider(sentences,
				sbjPosList, objPosList, labels);

		return new SentencePositionDataSetIterator.Builder().sentenceProvider(sentenceProvider).wordVectors(wordVectors)
				.positionVectors(positionVectors).POSVectors(POSVectors).minibatchSize(minibatchSize).maxSentenceLength(maxSentenceLength)
				.useNormalizedWordVectors(false).build();

	}
	
	public DataSetIterator getSingleDataSetIterator(boolean isTraining, WordVectors wordVectors,
			PositionVectors positionVectors, POSVectors POSVectors, int minibatchSize, int maxSentenceLength, Random rng,
			TokenizerFactory tokenizerFactory, List<String> testList) throws Exception {

		List<String> sentences = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		List<Integer> sbjPosList = new ArrayList<Integer>();
		List<Integer> objPosList = new ArrayList<Integer>();

		String input = null;
		
		for(String test : testList) {
			input = test;
			
			StringTokenizer st = new StringTokenizer(input, "\t");
			String stc = st.nextToken();
			int sbjPos = Integer.parseInt(st.nextToken());
			int objPos = Integer.parseInt(st.nextToken());
			String label = st.nextToken();

			String[] stcArr = stc.split(" ");
			if (stcArr.length > GlobalVariables.truncateSentencesToLength - 1) {
				continue;
			}

			if (sbjPos == objPos) {
				continue;
			}

			sentences.add(stc);
			labels.add(label);
			sbjPosList.add(sbjPos);
			objPosList.add(objPos);
			
		}
		
		LabeledSentencePositionProvider sentenceProvider = new CollectionLabeledSentencePositionProvider(sentences,
				sbjPosList, objPosList, labels);

		return new SentencePositionDataSetIterator.Builder().sentenceProvider(sentenceProvider).wordVectors(wordVectors)
				.positionVectors(positionVectors).minibatchSize(minibatchSize).maxSentenceLength(maxSentenceLength)
				.useNormalizedWordVectors(false).build();

	}

}
