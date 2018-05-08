package edu.kaist.mrlab.nn.pcnn.utilities;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

public class Test {
	public static void main(String[] ar) throws Exception {
		BufferedReader br = Files.newBufferedReader(Paths.get("data/gs/gold_standard_intersection"));

		Set<String> propertySet = new HashSet<String>();
		
		String input = null;
		while ((input = br.readLine()) != null) {

			StringTokenizer st = new StringTokenizer(input, "\t");
			st.nextToken();
			st.nextToken();
			String property = st.nextToken();
			
			propertySet.add(property);
			
		}
		
		for(String property : propertySet) {
			System.out.print(property + ", ");
		}
	}
}
