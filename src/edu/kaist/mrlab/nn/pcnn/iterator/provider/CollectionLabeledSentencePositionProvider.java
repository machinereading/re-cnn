package edu.kaist.mrlab.nn.pcnn.iterator.provider;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.datavec.api.util.RandomUtils;
import org.deeplearning4j.berkeley.Pair;

import edu.kaist.mrlab.nn.pcnn.iterator.LabeledSentencePositionProvider;
import lombok.NonNull;

/**
 * 
 * @author sangha
 *
 */
public class CollectionLabeledSentencePositionProvider implements LabeledSentencePositionProvider {

	private final List<String> sentences;
	private final List<Integer> sbjPos;
	private final List<Integer> objPos;
	private final List<String> labels;
	private final Random rng;
	private final int[] order;
	private final List<String> allLabels;

	private int cursor = 0;

	public CollectionLabeledSentencePositionProvider(@NonNull List<String> sentences, @NonNull List<Integer> sbjPos,
			@NonNull List<Integer> objPos, @NonNull List<String> labelsForSentences) {
		this(sentences, sbjPos, objPos, labelsForSentences, new Random());
	}

	public CollectionLabeledSentencePositionProvider(@NonNull List<String> sentences, @NonNull List<Integer> sbjPos,
			@NonNull List<Integer> objPos, @NonNull List<String> labelsForSentences, Random rng) {
		if (sentences.size() != labelsForSentences.size()) {
			throw new IllegalArgumentException("Sentences and labels must be same size (sentences size: "
					+ sentences.size() + ", labels size: " + labelsForSentences.size() + ")");
		}

		this.sentences = sentences;
		this.sbjPos = sbjPos;
		this.objPos = objPos;
		this.labels = labelsForSentences;
		this.rng = rng;
		if (rng == null) {
			order = null;
		} else {
			order = new int[sentences.size()];
			for (int i = 0; i < sentences.size(); i++) {
				order[i] = i;
			}

//			RandomUtils.shuffleInPlace(order, rng);
		}

		// Collect set of unique labels for all sentences
		Set<String> uniqueLabels = new HashSet<>();
		for (String s : labelsForSentences) {
			uniqueLabels.add(s);
		}
		allLabels = new ArrayList<>(uniqueLabels);
		Collections.sort(allLabels);
		
	}

	@Override
	public boolean hasNext() {
		return cursor < sentences.size();
	}

	@Override
	public Pair<String, String> nextSentence() {
		int idx;
		if (rng == null) {
			idx = cursor++;
		} else {
			idx = order[cursor++];
		}
		return new Pair<>(sentences.get(idx), labels.get(idx));
	}

	@Override
	public void reset() {
		cursor = 0;
		if (rng != null) {
			RandomUtils.shuffleInPlace(order, rng);
		}
	}

	@Override
	public int totalNumSentences() {
		return sentences.size();
	}

	@Override
	public List<String> allLabels() {
		return allLabels;
	}

	@Override
	public int numLabelClasses() {
		return allLabels.size();
	}

	@Override
	public String[] nextSentenceWithPosition() {
		String[] result = new String[4];

		int idx;
		if (rng == null) {
			idx = cursor++;
		} else {
			idx = order[cursor++];
		}

		result[0] = sentences.get(idx);
		result[1] = sbjPos.get(idx).toString();
		result[2] = objPos.get(idx).toString();
		result[3] = labels.get(idx);

		return result;
	}

}
