package edu.kaist.mrlab.nn.pcnn.postpro;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

import edu.kaist.mrlab.nn.pcnn.utilities.OntoProperty;

public class DummyRemover {
	public static void main(String[] ar) throws Exception {
		BufferedReader br = Files.newBufferedReader(Paths.get("data/test/pl4-out-orig"));
		BufferedWriter bw = Files.newBufferedWriter(Paths.get("data/test/pl4-out-intersect"));
		
		Set<String> propertySet = new HashSet<String>();
		for(OntoProperty p : OntoProperty.values()) {
			String property = p.toString();
			propertySet.add(property);
		}
		
		String input = null;
		while((input = br.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(input, "\t");
			String sbj = st.nextToken();
			String obj = st.nextToken();
			String label = st.nextToken();
			String predicted = st.nextToken();
			String score = st.nextToken();
			String stc = st.nextToken();
			
			if(propertySet.contains(label)) {
				bw.write(sbj + "\t" + obj + "\t" + label + "\t" + predicted + "\t" + score + "\n");
			}
		}
		
		bw.close();
		br.close();
	}
}
