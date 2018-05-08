package edu.kaist.mrlab.nn.pcnn.utilities;

import java.util.ArrayList;
import java.util.List;

public class SentenceStore {

	List<String> sentences;

	public SentenceStore() {
		this.sentences = new ArrayList<String>();
	}

	public void addSentence(String sentence) {
		this.sentences.add(sentence);
	}

	public String getSentence(int idx) {
		return this.sentences.get(idx);
	}

	public void clear() {
		this.sentences.clear();
	}

	public void remove(int idx) {
		this.sentences.remove(idx);
	}
	
	public int size() {
		return sentences.size();
	}

}
