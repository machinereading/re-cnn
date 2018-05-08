package edu.kaist.mrlab.nn.pcnn.iterator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import edu.kaist.mrlab.nn.pcnn.Learner;
import edu.kaist.mrlab.nn.pcnn.Testor;
import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;
import edu.kaist.mrlab.nn.pcnn.utilities.POSVectors;
import edu.kaist.mrlab.nn.pcnn.utilities.PositionVectors;
import edu.kaist.mrlab.nn.pcnn.utilities.SentencePositionUnit;

/**
 * 
 * @author sangha
 *
 */

public class SentencePositionDataSetIterator implements DataSetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public enum UnknownWordHandling {
		RemoveWord, UseUnknownVector
	}

	private static final String UNKNOWN_WORD_SENTINEL = "./Punctuation";
	// private static final String UNKNOWN_WORD_SENTINEL = ".";

	private LabeledSentencePositionProvider sentenceProvider = null;
	private WordVectors wordVectors;
	private PositionVectors positionVectors;
	private POSVectors POSVectors;
	@SuppressWarnings("unused")
	private TokenizerFactory tokenizerFactory;
	private UnknownWordHandling unknownWordHandling;
	private boolean useNormalizedWordVectors;
	private int minibatchSize;
	private int maxSentenceLength;
	private boolean sentencesAlongHeight;
	private DataSetPreProcessor dataSetPreProcessor;

	private int wordVectorSize;
	private int numClasses;
	private Map<String, Integer> labelClassMap;
	@SuppressWarnings("unused")
	private INDArray unknown;

	private int cursor = 0;

	private SentencePositionDataSetIterator(Builder builder) {

		this.sentenceProvider = builder.sentenceProvider;
		this.wordVectors = builder.wordVectors;
		this.positionVectors = builder.positionVectors;
		this.POSVectors = builder.POSVectors;
		this.tokenizerFactory = builder.tokenizerFactory;
		this.unknownWordHandling = builder.unknownWordHandling;
		this.useNormalizedWordVectors = builder.useNormalizedWordVectors;
		this.minibatchSize = builder.minibatchSize;
		this.maxSentenceLength = builder.maxSentenceLength;
		this.sentencesAlongHeight = builder.sentencesAlongHeight;
		this.dataSetPreProcessor = builder.dataSetPreProcessor;

		this.numClasses = this.sentenceProvider.numLabelClasses();
		this.labelClassMap = new HashMap<>();
		int count = 0;
		// First: sort the labels to ensure the same label assignment order (say
		// train vs. test)
		List<String> sortedLabels = new ArrayList<>(this.sentenceProvider.allLabels());
		Collections.sort(sortedLabels);

		for (String s : sortedLabels) {
			this.labelClassMap.put(s, count++);
		}
		Testor.labelClassMap = this.labelClassMap;
		if (unknownWordHandling == UnknownWordHandling.UseUnknownVector) {
			if (useNormalizedWordVectors) {
				wordVectors.getWordVectorMatrixNormalized(wordVectors.getUNK());
			} else {
				wordVectors.getWordVectorMatrix(wordVectors.getUNK());
			}
		}

		// + 2*5 to concat position vector + 5 POS Vector
		this.wordVectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length
				+ (2 * positionVectors.getPositionVectorSize());
		// + POSVectors.getPOVectorSize()
	}

	/**
	 * Generally used post training time to load a single sentence for predictions
	 */
	public INDArray loadSingleSentence(String sentence) {
		List<String> tokens = tokenizeSentence(sentence);

		int[] featuresShape = new int[] { 1, 1, 0, 0 };
		if (sentencesAlongHeight) {
			featuresShape[2] = Math.min(maxSentenceLength, tokens.size());
			featuresShape[3] = wordVectorSize;
		} else {
			featuresShape[2] = wordVectorSize;
			featuresShape[3] = Math.min(maxSentenceLength, tokens.size());
		}

		INDArray features = Nd4j.create(featuresShape);
		int length = (sentencesAlongHeight ? featuresShape[2] : featuresShape[3]);
		for (int i = 0; i < length; i++) {

			INDArray vector = getWordVector(tokens.get(i));

			INDArrayIndex[] indices = new INDArrayIndex[4];
			indices[0] = NDArrayIndex.point(0);
			indices[1] = NDArrayIndex.point(0);
			if (sentencesAlongHeight) {
				indices[2] = NDArrayIndex.point(i);
				indices[3] = NDArrayIndex.all();
			} else {
				indices[2] = NDArrayIndex.all();
				indices[3] = NDArrayIndex.point(i);
			}

			features.put(indices, vector);
		}

		return features;
	}

	private INDArray getWordVector(String word) {
		INDArray vector;
		if (unknownWordHandling == UnknownWordHandling.UseUnknownVector && word == UNKNOWN_WORD_SENTINEL) {
			// Yes, this *should* be using == for the sentinel String here
			// vector = unknown;
			vector = wordVectors.getWordVectorMatrixNormalized(word);
		} else {
			if (useNormalizedWordVectors) {
				vector = wordVectors.getWordVectorMatrixNormalized(word);
			} else {
				vector = wordVectors.getWordVectorMatrix(word);
			}
		}
		return vector;
	}

	private INDArray getPositionVector(int position) {
		INDArray vector = positionVectors.getPositionVectors(position);
		return vector;
	}

	private INDArray getPOSVector(String POS) {
		INDArray vector = POSVectors.getPOSVectors(POS);
		return vector;
	}

	private Pair<Integer, Integer> getRelativePosition(int idx, int sbjPos, int objPos) {

		int relSbj = idx - sbjPos;
		int relObj = idx - objPos;

		if (relSbj < -60) {
			relSbj = 0;
		} else if (relSbj <= 60 && relSbj >= -60) {
			relSbj += 61;
		} else if (relSbj > 60) {
			relSbj = 121;
		}

		if (relObj < -60) {
			relObj = 0;
		} else if (relObj <= 60 && relObj >= -60) {
			relObj += 61;
		} else if (relObj > 60) {
			relObj = 121;
		}

		return new Pair<>(relSbj, relObj);
	}

	private List<String> tokenizeSentence(String sentence) {
		// Tokenizer t = tokenizerFactory.create(sentence);
		StringTokenizer t = new StringTokenizer(sentence, " ");
		List<String> tokens = new ArrayList<>();
		while (t.hasMoreTokens()) {
			String token = t.nextToken();

			// for pos embedding
			// int slash = token.lastIndexOf("/");
			// token = token.substring(0, slash);
			/////////////

			if (!wordVectors.hasWord(token)) {
				switch (unknownWordHandling) {
				case RemoveWord:
					continue;
				case UseUnknownVector:
					token = UNKNOWN_WORD_SENTINEL;
				}
			}
			tokens.add(token);
		}
		return tokens;
	}

	private List<String> tokenizePOSSequence(String sentence) {
		// Tokenizer t = tokenizerFactory.create(sentence);
		StringTokenizer t = new StringTokenizer(sentence, " ");
		List<String> tokens = new ArrayList<>();
		while (t.hasMoreTokens()) {
			String token = t.nextToken();
			int slash = token.lastIndexOf("/");
			token = token.substring(slash + 1, token.length());
			tokens.add(token);
		}
		return tokens;
	}

	public Map<String, Integer> getLabelClassMap() {
		return new HashMap<>(labelClassMap);
	}

	@Override
	public boolean hasNext() {
		if (sentenceProvider == null) {
			throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
		}
		return sentenceProvider.hasNext();
	}

	@Override
	public DataSet next() {
		return next(minibatchSize);
	}

	@Override
	public DataSet next(int num) {
		if (sentenceProvider == null) {
			throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
		}

		/**
		 * List<String> == sentences String == label List<Pair<>> == training batch size
		 * num == training batch size (10)
		 */

		List<SentencePositionUnit> tokenizedSentencesWithPosition = new ArrayList<>(num);
		int maxLength = -1;
		int minLength = Integer.MAX_VALUE; // Track to we know if we can skip
											// mask creation for "all same
											// length" case
		for (int i = 0; i < num && sentenceProvider.hasNext(); i++) {
			String[] sp = sentenceProvider.nextSentenceWithPosition();
			String sentence = sp[0];
			List<String> tokens = tokenizeSentence(sentence);
			List<String> POSs = tokenizePOSSequence(sentence);
			if (tokens.size() == 0) {
				continue;
			}

			int count = 2;
			// for(String token : tokens) {
			// if(token.endsWith("/Entity")) {
			// count++;
			// }
			// }

			int sbjPos = Integer.parseInt(sp[1]);
			int objPos = Integer.parseInt(sp[2]);
			String label = sp[3];
			maxLength = Math.max(maxLength, tokens.size());

			if (count == 2) {
				tokenizedSentencesWithPosition.add(new SentencePositionUnit(tokens, sbjPos, objPos, label, POSs));

				if (sbjPos > GlobalVariables.truncateSentencesToLength - 1) {
					sbjPos = GlobalVariables.truncateSentencesToLength - 1;
				}
				if (objPos > GlobalVariables.truncateSentencesToLength - 1) {
					objPos = GlobalVariables.truncateSentencesToLength - 1;
				}

				Learner.positionStore.addSbjPos(sbjPos);
				Learner.positionStore.addObjPos(objPos);

				if (GlobalVariables.isTraining) {
					// pass
				} else {
					Testor.sentenceStore.addSentence(sentence);
					Testor.positionStore.addSbjPos(sbjPos);
					Testor.positionStore.addObjPos(objPos);
				}

			} else {
				// System.out.println(tokens);
			}

		}

		if (maxSentenceLength > 0 && maxLength > maxSentenceLength) {
			maxLength = maxSentenceLength;
		}

		// int currMinibatchSize = tokenizedSentences.size();
		int currMinibatchSize = tokenizedSentencesWithPosition.size();
		INDArray labels = Nd4j.create(currMinibatchSize, numClasses);
		for (int i = 0; i < tokenizedSentencesWithPosition.size(); i++) {
			String labelStr = tokenizedSentencesWithPosition.get(i).getLabel();
			if (!labelClassMap.containsKey(labelStr)) {
				throw new IllegalStateException(
						"Got label \"" + labelStr + "\" that is not present in list of LabeledSentenceProvider labels");
			}

			int labelIdx = labelClassMap.get(labelStr);

			labels.putScalar(i, labelIdx, 1.0);
		}

		int[] featuresShape = new int[4];
		featuresShape[0] = currMinibatchSize;
		featuresShape[1] = 1;
		if (sentencesAlongHeight) {
			featuresShape[2] = maxLength;
			featuresShape[3] = wordVectorSize;
		} else {
			featuresShape[2] = wordVectorSize;
			featuresShape[3] = maxLength;
		}

		INDArray features = Nd4j.create(featuresShape);
		for (int i = 0; i < currMinibatchSize; i++) {
			List<String> currSentence = tokenizedSentencesWithPosition.get(i).getTokens();
			// List<String> currPOS = tokenizedSentencesWithPosition.get(i).getPOSs();

			for (int j = 0; j < currSentence.size() && j < maxSentenceLength; j++) {

				Pair<Integer, Integer> relativePositionPair = getRelativePosition(j,
						tokenizedSentencesWithPosition.get(i).getSbjPos(),
						tokenizedSentencesWithPosition.get(i).getObjPos());

				INDArray sbjPosEmbeds = getPositionVector(relativePositionPair.getFirst());
				INDArray objPosEmbeds = getPositionVector(relativePositionPair.getSecond());
				INDArray wordVector = getWordVector(currSentence.get(j));
				// INDArray POSVector = getPOSVector(currPOS.get(j));

				// System.out.println(currSentence.get(j));
				// System.out.println(currPOS.get(j));
				// System.out.println(wordVector);
				// System.out.println(POSVector);
				// System.out.println(sbjPosEmbeds);
				// System.out.println(objPosEmbeds);
				// System.out.println();

				INDArray vector = Nd4j.concat(1, wordVector, sbjPosEmbeds, objPosEmbeds);
				// INDArray vector = Nd4j.concat(1, wordVector, sbjPosEmbeds, objPosEmbeds,
				// POSVector);

				INDArrayIndex[] indices = new INDArrayIndex[4];
				// TODO REUSE
				indices[0] = NDArrayIndex.point(i);
				indices[1] = NDArrayIndex.point(0);
				if (sentencesAlongHeight) {
					indices[2] = NDArrayIndex.point(j);
					indices[3] = NDArrayIndex.all();
				} else {
					indices[2] = NDArrayIndex.all();
					indices[3] = NDArrayIndex.point(j);
				}

				features.put(indices, vector);
			}
		}

		INDArray featuresMask = null;
		if (minLength != maxLength) {
			featuresMask = Nd4j.create(currMinibatchSize, maxLength);

			for (int i = 0; i < currMinibatchSize; i++) {
				int sentenceLength = tokenizedSentencesWithPosition.get(i).getTokens().size();
				if ((sentenceLength >= maxLength)) {
					featuresMask.getRow(i).assign(1.0);
				} else {
					featuresMask.get(NDArrayIndex.point(i), NDArrayIndex.interval(0, sentenceLength)).assign(1.0);
				}
			}
		}

		DataSet ds = new DataSet(features, labels, featuresMask, null);

		if (dataSetPreProcessor != null) {
			dataSetPreProcessor.preProcess(ds);
		}

		cursor += ds.numExamples();
		Learner.positionStore.setCursor(cursor);

		return ds;
	}

	@Override
	public int totalExamples() {
		return sentenceProvider.totalNumSentences();
	}

	@Override
	public int inputColumns() {
		return wordVectorSize;
	}

	@Override
	public int totalOutcomes() {
		return numClasses;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public void reset() {
		cursor = 0;
		sentenceProvider.reset();
	}

	@Override
	public int batch() {
		return minibatchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		this.dataSetPreProcessor = preProcessor;
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		return dataSetPreProcessor;
	}

	@Override
	public List<String> getLabels() {
		// We don't want to just return the list from the
		// LabelledSentenceProvider, as we sorted them earlier to do the
		// String -> Integer mapping
		String[] str = new String[labelClassMap.size()];
		for (Map.Entry<String, Integer> e : labelClassMap.entrySet()) {
			str[e.getValue()] = e.getKey();
		}
		return Arrays.asList(str);
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException("Not supported");
	}

	public static class Builder {
		private LabeledSentencePositionProvider sentenceProvider = null;
		private WordVectors wordVectors;
		private PositionVectors positionVectors;
		private POSVectors POSVectors;
		private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		private UnknownWordHandling unknownWordHandling = UnknownWordHandling.UseUnknownVector;
		private boolean useNormalizedWordVectors = true;
		private int maxSentenceLength = -1;
		private int minibatchSize = 32;
		private boolean sentencesAlongHeight = true;
		private DataSetPreProcessor dataSetPreProcessor;

		/**
		 * Specify how the (labelled) sentences / documents should be provided
		 */
		public Builder sentenceProvider(LabeledSentencePositionProvider labeledSentenceProvider) {
			this.sentenceProvider = labeledSentenceProvider;
			return this;
		}

		/**
		 * Provide the WordVectors instance that should be used for training
		 */
		public Builder wordVectors(WordVectors wordVectors) {
			this.wordVectors = wordVectors;
			return this;
		}

		/**
		 * Provide the PositionVectors instance that should be used for training
		 */
		public Builder positionVectors(PositionVectors positionVectors) {
			this.positionVectors = positionVectors;
			return this;
		}

		/**
		 * Provide the POSVectors instance that should be used for training
		 */
		public Builder POSVectors(POSVectors POSVectors) {
			this.POSVectors = POSVectors;
			return this;
		}

		/**
		 * The {@link TokenizerFactory} that should be used. Defaults to
		 * {@link DefaultTokenizerFactory}
		 */
		public Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
			this.tokenizerFactory = tokenizerFactory;
			return this;
		}

		/**
		 * Specify how unknown words (those that don't have a word vector in the
		 * provided WordVectors instance) should be handled. Default: remove/ignore
		 * unknown words.
		 */
		public Builder unknownWordHandling(UnknownWordHandling unknownWordHandling) {
			this.unknownWordHandling = unknownWordHandling;
			return this;
		}

		/**
		 * Minibatch size to use for the DataSetIterator
		 */
		public Builder minibatchSize(int minibatchSize) {
			this.minibatchSize = minibatchSize;
			return this;
		}

		/**
		 * Whether normalized word vectors should be used. Default: true
		 */
		public Builder useNormalizedWordVectors(boolean useNormalizedWordVectors) {
			this.useNormalizedWordVectors = useNormalizedWordVectors;
			return this;
		}

		/**
		 * Maximum sentence/document length. If sentences exceed this, they will be
		 * truncated to this length by taking the first 'maxSentenceLength' known words.
		 */
		public Builder maxSentenceLength(int maxSentenceLength) {
			this.maxSentenceLength = maxSentenceLength;
			return this;
		}

		/**
		 * If true (default): output features data with shape [minibatchSize, 1,
		 * maxSentenceLength, wordVectorSize]<br>
		 * If false: output features with shape [minibatchSize, 1, wordVectorSize,
		 * maxSentenceLength]
		 */
		public Builder sentencesAlongHeight(boolean sentencesAlongHeight) {
			this.sentencesAlongHeight = sentencesAlongHeight;
			return this;
		}

		/**
		 * Optional DataSetPreProcessor
		 */
		public Builder dataSetPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
			this.dataSetPreProcessor = dataSetPreProcessor;
			return this;
		}

		public SentencePositionDataSetIterator build() {
			if (wordVectors == null) {
				throw new IllegalStateException(
						"Cannot build CnnSentenceDataSetIterator without a WordVectors instance");
			}

			return new SentencePositionDataSetIterator(this);
		}

	}

}