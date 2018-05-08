package edu.kaist.mrlab.nn.pcnn.utilities;

public class TripleUnit {
	private String sbj;
	private String predicted;
	private String obj;
	private double score;
	private String sentence;
	private String label;
	
	public TripleUnit(String sbj, String predicted, String obj, double score, String sentence, String label){
		this.sbj = sbj;
		this.predicted = predicted;
		this.obj = obj;
		this.score = score;
		this.sentence = sentence;
		this.label = label;
	}

	public String getSbj() {
		return sbj;
	}

	public void setSbj(String sbj) {
		this.sbj = sbj;
	}

	public String getPredicted() {
		return predicted;
	}

	public void setPredicted(String predicted) {
		this.predicted = predicted;
	}

	public String getObj() {
		return obj;
	}

	public void setObj(String obj) {
		this.obj = obj;
	}

	public double getScore() {
		return score;
	}

	public void setScore(double score) {
		this.score = score;
	}

	public String getSentence() {
		return sentence;
	}

	public void setSentence(String sentence) {
		this.sentence = sentence;
	}

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}
	
	
	

}
