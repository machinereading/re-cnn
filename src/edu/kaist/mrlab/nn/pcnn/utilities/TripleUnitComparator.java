package edu.kaist.mrlab.nn.pcnn.utilities;

import java.util.Comparator;

public class TripleUnitComparator implements Comparator<TripleUnit> {

	@Override
	public int compare(TripleUnit o1, TripleUnit o2) {
		// TODO Auto-generated method stub
		double score1 = o1.getScore();
		double score2 = o2.getScore();
		if (score2 > score1) {
			return 1;
		} else if (score1 > score2) {
			return -1;
		} else {
			return 0;
		}
	}

}
