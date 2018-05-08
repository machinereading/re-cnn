package edu.kaist.mrlab.nn.pcnn.utilities;

public class TestUnit {
	private String sentence;
	private String label;
	private double score;
	private int sbjPos;
	private int objPos;

	public TestUnit(String sentence, String label, double score, int sbjPos, int objPos) {
		this.sentence = sentence;
		this.label = label;
		this.score = score;
		this.sbjPos = sbjPos;
		this.objPos = objPos;
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

	public double getScore() {
		return score;
	}

	public void setScore(double score) {
		this.score = score;
	}

	public int getSbjPos() {
		return sbjPos;
	}

	public void setSbjPos(int sbjPos) {
		this.sbjPos = sbjPos;
	}

	public int getObjPos() {
		return objPos;
	}

	public void setObjPos(int objPos) {
		this.objPos = objPos;
	}

}
