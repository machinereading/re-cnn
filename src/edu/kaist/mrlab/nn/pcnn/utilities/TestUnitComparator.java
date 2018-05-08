package edu.kaist.mrlab.nn.pcnn.utilities;

import java.util.Comparator;

public class TestUnitComparator implements Comparator<TestUnit> {

	@Override
	public int compare(TestUnit o1, TestUnit o2) {
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
