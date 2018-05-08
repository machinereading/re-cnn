package edu.kaist.mrlab.nn.pcnn.eval;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import edu.kaist.mrlab.nn.pcnn.Testor;
import edu.kaist.mrlab.nn.pcnn.utilities.Label;
import edu.kaist.mrlab.nn.pcnn.utilities.OntoProperty;
import edu.kaist.mrlab.nn.pcnn.utilities.TripleUnit;
import edu.kaist.mrlab.nn.pcnn.utilities.TripleUnitComparator;

public class ComputationGraphEvaluator {

	private ComputationGraph net;

	public ComputationGraphEvaluator(ComputationGraph net) {
		this.net = net;
	}

	/**
	 * Evaluate the network (classification performance)
	 *
	 * @param iterator
	 *            Iterator to evaluate on
	 * @return Evaluation object; results of evaluation on all examples in the data
	 *         set
	 */

	public Evaluation evaluate(DataSetIterator iterator) throws Exception {
		return evaluate(iterator, null);
	}

	/**
	 * Evaluate the network on the provided data set. Used for evaluating the
	 * performance of classifiers
	 *
	 * @param iterator
	 *            Data to undertake evaluation on
	 * @return Evaluation object, summarizing the results of the evaluation on the
	 *         provided DataSetIterator
	 */
	public Evaluation evaluate(DataSetIterator iterator, List<String> labelsList) throws Exception {
		return evaluate(iterator, labelsList, 1);
	}

	/**
	 * Evaluate the network (for classification) on the provided data set, with top
	 * N accuracy in addition to standard accuracy. For 'standard' accuracy
	 * evaluation only, use topN = 1
	 *
	 * @param iterator
	 *            Iterator (data) to evaluate on
	 * @param labelsList
	 *            List of labels. May be null.
	 * @param topN
	 *            N value for top N accuracy evaluation
	 * @return Evaluation object, summarizing the results of the evaluation on the
	 *         provided DataSetIterator
	 */
	public Evaluation evaluate(DataSetIterator iterator, List<String> labelsList, int topN) throws Exception {

		// BufferedWriter bw = Files
		// .newBufferedWriter(Paths.get("/test/wjd1004109/PL-Web-Demo/DB2XB/data/pl_out/pl4-out"));

		BufferedWriter bw = Files.newBufferedWriter(Paths.get("data/test/pl4-out-orig"));
		BufferedWriter bw2 = Files.newBufferedWriter(Paths.get("data/test/pl4-out-all"));

		// BufferedWriter bw = Files
		// .newBufferedWriter(Paths.get("data/test/ko_test_result.txt"));

		List<TripleUnit> tripleList = new ArrayList<TripleUnit>();
		List<TripleUnit> allTripleList = new ArrayList<TripleUnit>();

		if (labelsList == null)
			labelsList = iterator.getLabels();

		labelsList = new ArrayList<String>();
		for (Label p : Label.values()) {
			String label = p.toString();
			if (label.equals("classP")) {
				label = "class";
			}
			labelsList.add(label);
		}

		List<String> sortedLabels = new ArrayList<String>();
		for (OntoProperty p : OntoProperty.values()) {
			String property = p.toString();
			if (property.equals("classP")) {
				property = "class";
			}
			sortedLabels.add(property);
		}
		Collections.sort(sortedLabels);

		int count = 0;
		Map<String, Integer> labelClassMap = new HashMap<>();
		for (String s : sortedLabels) {
			labelClassMap.put(s, count++);
		}

		Map<Integer, String> propertyLabelMap = new HashMap<>();
		for (String property : labelClassMap.keySet()) {
			int propertyID = labelClassMap.get(property);
			propertyLabelMap.put(propertyID, property);
		}

		Evaluation e = new Evaluation(labelsList, topN);
		while (iterator.hasNext()) {
			org.nd4j.linalg.dataset.DataSet next = iterator.next();

			if (next.getFeatureMatrix() == null || next.getLabels() == null)
				break;

			INDArray features = next.getFeatures();
			INDArray labels = next.getLabels();

			INDArray[] out;
			out = net.output(false, features);

			INDArray guesses = out[0];
			INDArray guessIndex = Nd4j.argMax(guesses, 1);
			INDArray realOutcomeIndex = Nd4j.argMax(labels, 1);

			// int nExamples = guessIndex.length();
			// for (int i = 0; i < nExamples; i++) {
			// int predicted = (int) guessIndex.getDouble(i);
			// int actual = (int) realOutcomeIndex.getDouble(i);
			//
			// String answer = propertyLabelMap.get(predicted);
			// String label = propertyLabelMap.get(actual);
			//
			// tripleList.add(new TripleUnit("", answer, "", 0, "", label));
			// }

			for (int i = 0; i < guesses.rows(); i++) {

				realOutcomeIndex = Nd4j.argMax(labels.getRow(i), 1);
				int actual = (int) realOutcomeIndex.getDouble(0);
				String answer = propertyLabelMap.get(actual);

				String sentence = Testor.sentenceStore.getSentence(i);
				int sbjPos = Testor.positionStore.getSbjPos().get(i);
				int objPos = Testor.positionStore.getObjPos().get(i);
				INDArray temp = guesses.getRow(i);
				INDArray guessScore = temp.max(1);
				guessIndex = Nd4j.argMax(temp, 1);
				int predictedIdx = (int) guessIndex.getDouble(0);
				double predictedScore = guessScore.getDouble(0);
				String predicted = propertyLabelMap.get(predictedIdx);
				String[] tokens = sentence.split(" ");
				String sbj = tokens[sbjPos].replace("/Entity", "");
				String obj = tokens[objPos].replace("/Entity", "");
				
				String origSentence = null;
				if(Testor.originalSentenceStore.size() > 0) {
					origSentence = Testor.originalSentenceStore.getSentence(0);
					Testor.originalSentenceStore.remove(0);
				}

				for (int j = 0; j < temp.length(); j++) {
					double idxScore = temp.getDouble(j);
					String idxLabel = propertyLabelMap.get(j);
					allTripleList.add(new TripleUnit(sbj, idxLabel, obj, idxScore, origSentence, answer));
				}

				tripleList.add(new TripleUnit(sbj, predicted, obj, predictedScore, origSentence, answer));
			}

			Testor.sentenceStore.clear();
			Testor.positionStore.clear();

		}

		TripleUnitComparator comparator = new TripleUnitComparator();
		Collections.sort(tripleList, comparator);


		for (TripleUnit tu : tripleList) {

			bw.write(tu.getSbj() + "\t" + tu.getPredicted() + "\t" + tu.getObj() + "\t.\t" + tu.getScore() + "\t"
					+ tu.getSentence() + "\n");

		}

		for (TripleUnit tu : allTripleList) {
			bw2.write(tu.getSbj() + "\t" + tu.getPredicted() + "\t" + tu.getObj() + "\t.\t" + tu.getScore() + "\t"
					+ tu.getSentence() + "\n");
		}

		bw.close();
		bw2.close();
		return e;
	}

}
