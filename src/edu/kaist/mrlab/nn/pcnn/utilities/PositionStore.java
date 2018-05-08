package edu.kaist.mrlab.nn.pcnn.utilities;

import java.util.ArrayList;
import java.util.List;

/**
 * 
 * @author sangha
 *
 */
public class PositionStore {

	private List<Integer> sbjPosList;
	private List<Integer> objPosList;

	private int cursor;

	public PositionStore() {

		this.sbjPosList = new ArrayList<Integer>();
		this.objPosList = new ArrayList<Integer>();

	}

	public void addSbjPos(int sbjPos) {
		this.sbjPosList.add(sbjPos);
	}

	public void addObjPos(int objPos) {
		this.objPosList.add(objPos);
	}

	public List<Integer> getSbjPos(int cursor) {
		
		if(cursor + 8 > this.sbjPosList.size()){
			return this.sbjPosList.subList(cursor, this.sbjPosList.size());
		}
		return this.sbjPosList.subList(cursor, cursor + GlobalVariables.batchSize);
	}

	public List<Integer> getObjPos(int cursor) {
		
		if(cursor + 8 > this.objPosList.size()){
			return this.objPosList.subList(cursor, this.objPosList.size());
		}
		return this.objPosList.subList(cursor, cursor + GlobalVariables.batchSize);
	}

	public List<Integer> getSbjPos() {
		return this.sbjPosList;
	}

	public List<Integer> getObjPos() {
		return this.objPosList;
	}

	public int getCursor() {
		return cursor;
	}

	public void setCursor(int cursor) {
		this.cursor = cursor;
	}
	
	public void clear(){
		this.sbjPosList.clear();
		this.objPosList.clear();
	}

}
