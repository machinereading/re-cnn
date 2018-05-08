package edu.kaist.mrlab.nn.pcnn;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;

public class ModelTestor {
	public static void main(String[] ar) throws Exception {

		HashMap<String, Integer> labelTotal = new HashMap<>();
		HashMap<String, Integer> extrTotal = new HashMap<>();
		HashMap<String, Integer> truePositive = new HashMap<>();

		BufferedReader br = Files.newBufferedReader(Paths.get("data/test/model-test"));
		String input = null;
		while ((input = br.readLine()) != null) {
			input = input.replaceAll(":", "");
			String[] tokens = input.split(" ");
			String label = tokens[3];
			String extr = tokens[8];
			int count = Integer.parseInt(tokens[9]);

			if (labelTotal.containsKey(label)) {
				int totalCount = labelTotal.get(label);
				totalCount += count;
				labelTotal.put(label, totalCount);
			} else {
				labelTotal.put(label, count);
			}

			if (extrTotal.containsKey(extr)) {
				int totalCount = extrTotal.get(extr);
				totalCount += count;
				extrTotal.put(extr, totalCount);
			} else {
				extrTotal.put(extr, count);
			}

			if (label.equals(extr)) {
				truePositive.put(label, count);
			}
		}

		System.out.println("property\tprecision\trecall\tf1");
		Iterator<?> it = truePositive.keySet().iterator();
		while (it.hasNext()) {
			String property = it.next().toString();
			int truePositiveCount = truePositive.get(property);
			int labelCount = labelTotal.get(property);
			int extrCount = extrTotal.get(property);

			double recall = Double.parseDouble(String.format("%.2f", (truePositiveCount / (double) labelCount)));
			double precision = Double.parseDouble(String.format("%.2f", (truePositiveCount / (double) extrCount)));
			double f1 = Double.parseDouble(String.format("%.2f", (2 * precision * recall / (precision + recall))));

			
			System.out.println(property + "\t" + precision + "\t" + recall + "\t" + f1);
		}
	}
}
