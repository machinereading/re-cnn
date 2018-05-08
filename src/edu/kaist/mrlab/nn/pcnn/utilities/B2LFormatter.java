package edu.kaist.mrlab.nn.pcnn.utilities;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.StringTokenizer;

public class B2LFormatter {
	public static void main(String[] ar) throws Exception {
		BufferedReader br = Files
				.newBufferedReader(Paths.get("data/ds/dbp-wiki/4_ds_labeled_data_prop_section_New_ko_over1000"));
		BufferedWriter bw = Files
				.newBufferedWriter(Paths.get("data/ds/dbp-wiki/kowiki-20170701-dbpedia-wikilink-sentence-extended"));

		HashSet<String> output = new HashSet<>();
		HashSet<String> predSet = new HashSet<>();

		String input = null;
		while ((input = br.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(input, "\t");
			String stc = st.nextToken();
			String sbj = st.nextToken();
			String obj = st.nextToken();
			String pred = st.nextToken();
			String origStc = stc;
			
			predSet.add(pred);

			if (!stc.contains("HeadEntity") || !stc.contains("TailEntity")) {
				continue;
			}

			int pivot = 0;
			ArrayList<Integer> tailEntityPosition = new ArrayList<>();
			int tPosition = stc.indexOf("TailEntity", pivot);
			tailEntityPosition.add(tPosition);
			pivot = tPosition + 1;
			while (tPosition != -1) {
				tPosition = stc.indexOf("TailEntity", pivot);
				pivot = tPosition + 1;
				if (tPosition == -1) {
					continue;
				}
				tailEntityPosition.add(tPosition);
			}

			pivot = 0;
			ArrayList<Integer> headEntityPosition = new ArrayList<>();
			int hPosition = stc.indexOf("HeadEntity", pivot);
			headEntityPosition.add(hPosition);
			pivot = hPosition + 1;
			while (hPosition != -1) {
				hPosition = stc.indexOf("HeadEntity", pivot);
				pivot = hPosition + 1;
				if (hPosition == -1) {
					continue;
				}
				headEntityPosition.add(hPosition);
			}

			for (int tailPosition : tailEntityPosition) {

				for (int headPosition : headEntityPosition) {

					String first = origStc.substring(0, tailPosition);
					String second = origStc.substring(tailPosition + 10, origStc.length());

					stc = first + "OBJECTREPL" + second;

					first = stc.substring(0, headPosition);
					second = stc.substring(headPosition + 10, stc.length());

					stc = first + "SBJECTREPL" + second;

					stc = stc.replace("HeadEntity", sbj);
					stc = stc.replace("TailEntity", obj);

					stc = stc.replace("OBJECTREPL", " << _obj_ >> ");
					stc = stc.replace("SBJECTREPL", " << _sbj_ >> ");

					output.add(sbj + "\t" + obj + "\t" + pred + "\t" + stc);

				}

			}

		}
		
		Iterator<String> predIt = predSet.iterator();
		while(predIt.hasNext()) {
			System.out.print(predIt.next() + ", ");
		}

		Iterator<String> it = output.iterator();
		while (it.hasNext()) {
			bw.write(it.next() + "\n");
		}

		br.close();
		bw.close();
	}
}
