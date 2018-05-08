package edu.kaist.mrlab.nn.pcnn.utilities;

import java.util.List;

/**
 * 
 * @author sangha
 *
 */
public class SentencePositionUnit {
	List<String> tokens;
	int sbjPos;
	int objPos;
	String label;
	// for pos embedding
	List<String> POSs;

	public SentencePositionUnit(List<String> tokens, int sbjPos, int objPos, String label, List<String> POSs) {
		this.tokens = tokens;
		this.sbjPos = sbjPos;
		this.objPos = objPos;
		this.label = label;
		this.POSs = POSs;
	}

	public List<String> getTokens() {
		return tokens;
	}

	public void setTokens(List<String> tokens) {
		this.tokens = tokens;
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

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

	public List<String> getPOSs() {
		return POSs;
	}

	public void setPOSs(List<String> pOSs) {
		POSs = pOSs;
	}
	
	

}
