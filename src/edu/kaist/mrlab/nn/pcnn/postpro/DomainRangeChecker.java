package edu.kaist.mrlab.nn.pcnn.postpro;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import edu.kaist.mrlab.nn.pcnn.utilities.PropertyDR;

public class DomainRangeChecker {
	private Map<String, Set<String>> entityType = new HashMap<>();
	private Map<String, PropertyDR> propertySet = new HashMap<>();
	private Map<String, Integer> entityCount = new HashMap<>();

	public void loadEntityCount() throws Exception {
		BufferedReader br = Files.newBufferedReader(Paths.get("data/entity_prop_kbox"));
		String input = null;
		while ((input = br.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(input, "\t");
			String entity = st.nextToken();
			int eCount = Integer.parseInt(st.nextToken());
			entityCount.put(entity, eCount);
		}
	}

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

			PropertyDR p = new PropertyDR(property);
			p.setDomain(domain);
			p.setRange(range);
			propertySet.put(property, p);
		}
	}

	public String filter() throws Exception {
		
		String result = "data/test/pl4-out";
		
		BufferedReader br = Files.newBufferedReader(Paths.get("data/test/pl4-out-orig"));
		BufferedWriter bw = Files.newBufferedWriter(Paths.get(result));
		// BufferedWriter bw = Files
		// .newBufferedWriter(Paths.get("/test/wjd1004109/PL-Web-Demo/DB2XB/data/pl_out/pl4-out"));

		String input = null;
		while ((input = br.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(input, "\t");
			String sbj = st.nextToken();
			String pred = st.nextToken();
			String obj = st.nextToken();
			String dot = st.nextToken();
			double score = Double.parseDouble(st.nextToken());
			String stc = st.nextToken();

			Set<String> sbjType = entityType.get(sbj);
			Set<String> objType = entityType.get(obj);

			// System.out.println(pred);
			PropertyDR property = propertySet.get(pred);
			String domain = property.getDomain();
			String range = property.getRange();

			// if(score > 0.6) {
			// System.out.println(sbj + "\t" + pred + "\t" + obj + "\t" + dot + "\t" + score
			// + "\t" + stc);
			// }

			// sbjType == null || domain == null ||
			// objType == null || range == null ||

			boolean isPassable = true;

			if (!domain.equals("null") && (sbjType != null && !sbjType.contains(domain))) {
				isPassable = false;
			}

			if (!range.equals("null") && (objType != null && !objType.contains(range))) {
				isPassable = false;
			}

			if (entityCount.containsKey(sbj)) {
				if (entityCount.get(sbj) < 1) {
					isPassable = false;
				}
			} else {
				isPassable = false;
			}

			if (entityCount.containsKey(obj)) {
				if (entityCount.get(obj) < 1) {
					isPassable = false;
				}
			} else {
				isPassable = false;
			}

			if (isPassable) {

				if (sbjType == null) {
					score = score * 0.8;
				}

				if (objType == null) {
					score = score * 0.8;
				}

				if (domain.equals("null")) {
					score = score * 0.8;
				}

				if (range.equals("null")) {
					score = score * 0.8;
				}

				bw.write(sbj + "\t" + pred + "\t" + obj + "\t" + dot + "\t" + score + "\t" + stc + "\n");

				System.out.println(sbj + "\t" + pred + "\t" + obj + "\t" + dot + "\t" + score + "\t" + stc);
			}

		}
		br.close();
		bw.close();
		
		return result;
	}

	public static void main(String[] ar) throws Exception {

		DomainRangeChecker drc = new DomainRangeChecker();
		drc.loadEntityType();
		drc.loadPropertyDomainRange();
		drc.loadEntityCount();
		drc.filter();

	}

}
