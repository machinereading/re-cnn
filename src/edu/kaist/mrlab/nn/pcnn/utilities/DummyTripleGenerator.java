package edu.kaist.mrlab.nn.pcnn.utilities;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import akka.japi.Pair;

public class DummyTripleGenerator {

	private Map<String, Set<String>> entityType = new HashMap<>();
	private Map<String, Property> propertySet = new HashMap<>();

	int maxSentenceLength = -1;

	public void loadEntityType() throws Exception {
		BufferedReader br = Files.newBufferedReader(Paths.get("data/entity_type_kbox"));
		String input = null;
		while ((input = br.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(input, "\t");
			String entity = st.nextToken();
			String eType = st.nextToken();
			if (entityType.containsKey(entity)) {
				Set<String> eTypeSet = entityType.get(entity);
				eTypeSet.add(eType);
				entityType.remove(entity);
				entityType.put(entity, eTypeSet);
			} else {
				HashSet<String> eTypeSet = new HashSet<>();
				eTypeSet.add(eType);
				entityType.put(entity, eTypeSet);
			}
		}
	}

	public void loadPropertyDomainRange() throws Exception {
		BufferedReader br = Files.newBufferedReader(Paths.get("data/all_property_domain_range"));
		String input = null;
		while ((input = br.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(input, "\t");
			String property = st.nextToken();
			String domain = st.nextToken();
			String range = st.nextToken();

			Property p = new Property(property);
			p.setDomain(domain);
			p.setRange(range);
			propertySet.put(property, p);
		}
	}

	public List<Pair<String, String>> getEntityPairs(String input) {

		List<String> entities = new ArrayList<String>();

		StringTokenizer st = new StringTokenizer(input, " ");
		boolean isEntity = false;
		String entity = "";
		while (st.hasMoreTokens()) {
			String token = st.nextToken();
			if (token.equals("<<")) {
				isEntity = true;
				continue;
			} else if (token.equals(">>")) {
				isEntity = false;
				entities.add(entity);
				entity = "";
			}
			if (isEntity) {
				entity += token;
			}
		}
		return shuffle(entities);
	}

	public List<Pair<String, String>> shuffle(List<String> entities) {

		List<Pair<String, String>> entityPairs = new ArrayList<>();

		for (int i = 0; i < entities.size(); i++) {
			for (int j = i + 1; j < entities.size(); j++) {

				String e1 = entities.get(i);
				String e2 = entities.get(j);
				Pair<String, String> entityPair = Pair.create(e1, e2);
				entityPairs.add(entityPair);
				Pair<String, String> entityPairInv = Pair.create(e2, e1);
				entityPairs.add(entityPairInv);

			}
		}

		return entityPairs;
	}

	public void generate() throws Exception {

		int stcCount = 0;
		int tripleCount = 0;
		int paragraphCount = 0;

		BufferedReader br = Files.newBufferedReader(Paths.get("data/gs/prior_gs_paragraph.txt"));
		BufferedWriter bw = Files.newBufferedWriter(Paths.get("data/gs/prior_gs_triple_strict"));

		// String pageTitle = br.readLine();
		// bw.write(pageTitle + "\n\n");

		String input = null;
		String paragraph = "";

		String prevTitle = "";
		String prevSubtitle = "";
		while ((input = br.readLine()) != null) {

			stcCount++;

			if (input.length() == 0) {
				continue;
			}

			String arr[] = input.split(":-:");
			String stc = arr[arr.length - 1];
			String title = arr[0];
			String subtitle = arr[1];

			if (stcCount == 1) {
				prevTitle = title;
				prevSubtitle = subtitle;
				paragraph += stc + " ";
				continue;
			} else if (title.equals(prevTitle) && subtitle.equals(prevSubtitle)) {
				prevTitle = title;
				prevSubtitle = subtitle;
				paragraph += stc + " ";
				continue;
			} else {

				paragraphCount++;

				List<Pair<String, String>> entityPairs = getEntityPairs(paragraph);

				for (int i = 0; i < entityPairs.size(); i++) {
					Pair<String, String> entityPair = entityPairs.get(i);
					String e1 = entityPair.first();
					String e2 = entityPair.second();

					if (e1.equals(e2) || e1.length() == 0 || e2.length() == 0) {
						continue;
					}

					Set<String> sbjType = entityType.get(e1);
					Set<String> objType = entityType.get(e2);

					for (OntoProperty ontoP : OntoProperty.values()) {

						boolean passable = true;

						String p = ontoP.toString();
						if (p.equals("classP")) {
							p = "class";
						}
						// System.out.println(p);
						Property property = propertySet.get(p);
						String domain = property.getDomain();
						String range = property.getRange();

						// if (sbjType == null || objType == null || domain == null || range == null) {
						// continue;
						// }
						

						if (!domain.equals("null") && sbjType != null) {
							if (!sbjType.contains(domain)) {
								passable = false;
							}
						}

						if (!range.equals("null") && objType != null) {
							if (!objType.contains(range)) {
								passable = false;
							}
						}
						
						if(domain.equals("null") || range.equals("null")) {
							passable = false;
						}

						// if ((sbjType.contains(domain)) && (objType.contains(range))) {
						if (passable) {
							String writeStc = paragraph.replace(" << " + e1 + " >> ", " << _sbj_ >> ");
							writeStc = writeStc.replace(" << " + e2 + " >> ", " << _obj_ >> ");

							if (maxSentenceLength < writeStc.length()) {
								maxSentenceLength = writeStc.length();
							}

							if (writeStc.length() < 8000) {
								tripleCount++;
								bw.write(e1 + "\t" + e2 + "\t" + property.getProperty() + "\t" + writeStc + "\n");
							}
						}
					}

					// Iterator<String> propIt = propertySet.keySet().iterator();
					// while (propIt.hasNext()) {
					// Property property = propertySet.get(propIt.next());
					// String domain = property.getDomain();
					// String range = property.getRange();
					//
					// if (sbjType == null || objType == null || domain == null || range == null) {
					// continue;
					// }
					// if ((sbjType.contains(domain)) && (objType.contains(range))) {
					// tripleCount++;
					//
					// String writeStc = paragraph.replace(" << " + e1 + " >> ", " << _sbj_ >> ");
					// writeStc = writeStc.replace(" << " + e2 + " >> ", " << _obj_ >> ");
					//
					// bw.write(e1 + "\t" + e2 + "\t" + property.getProperty() + "\t" + writeStc +
					// "\n");
					// }
					// }
				}

				paragraph = "";
				stcCount = 0;
			}
		}
		System.out.println("paragarph : " + paragraphCount);
		System.out.println("triple : " + tripleCount);
		System.out.println("maxSentenceLength : " + maxSentenceLength);

		br.close();
		bw.close();
	}

	public Set<String> getCombination(List<String> entitiesInSentence) {
		Set<String> combination = new HashSet<>();

		for (int i = 0; i < entitiesInSentence.size(); i++) {
			for (int j = i + 1; j < entitiesInSentence.size(); j++) {

				if (i == j) {
					continue;
				}

				String combi = entitiesInSentence.get(i) + "\t" + entitiesInSentence.get(j);
				String invConbi = entitiesInSentence.get(j) + "\t" + entitiesInSentence.get(i);
				combination.add(combi);
				combination.add(invConbi);

			}
		}

		return combination;
	}

	public static void main(String[] ar) throws Exception {

		DummyTripleGenerator dtg = new DummyTripleGenerator();
		dtg.loadEntityType();
		dtg.loadPropertyDomainRange();
		dtg.generate();
	}
}
