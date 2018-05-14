package edu.kaist.mrlab.nn.pcnn;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;

public class Configuration {

	/**
	 * If semeval file is put, set TRUE
	 */
	public static boolean isSemeval;

	/**
	 * The root directory containing input files
	 */
	public static String root;

	/**
	 * Semeval-style training input file path
	 */
	public static String semevalFile;

	/**
	 * KAIST-DS-style training input file path
	 */
	public static String dsInputFilePath;

	/**
	 * Path of Korean POS Parsed (Twitter) files
	 */
	public static String twitterParsedFolder;

	/**
	 * Path of Train/Dev. files
	 */
	public static String trainDevFolder;

	/**
	 * Development set ratio (default, 0.1)
	 */
	public static double devRatio;

	/**
	 * Semeval-style test file input
	 */
	public static String semevalTestFile;

	/**
	 * KAIST-DS-style test file input
	 */
	public static String testInputFilePath;

	/**
	 * Path of Testing files
	 */
	public static String testFolder;

	/**
	 * Testing set ratio (default, 1.0)
	 */
	public static double testRatio;

	/**
	 * If entity-tagged sentences is put, set TRUE
	 */
	public static boolean isLive;

	public static void load(Path configFile) throws Exception {
		BufferedReader br = Files.newBufferedReader(configFile);
		String input = null;
		while ((input = br.readLine()) != null) {

			if (input.trim().isEmpty() || input.startsWith("#")) {
				continue;
			}

			String[] kv = input.split("\\s+");
			switch (kv[0]) {
			case "isSemeval":
				String isS = kv[1].trim();
				if(isS.equals("TRUE")) {
					Configuration.isSemeval = true;
				} else {
					Configuration.isSemeval = false;
				}
				break;
			case "root":
				Configuration.root = kv[1].trim();
				break;
			case "semevalFile":
				Configuration.semevalFile = kv[1].trim();
				break;
			case "dsInputFilePath":
				Configuration.dsInputFilePath = kv[1].trim();
				break;
			case "twitterParsedFolder":
				Configuration.twitterParsedFolder = kv[1].trim();
				break;
			case "trainDevFolder":
				Configuration.trainDevFolder = kv[1].trim();
				break;
			case "devRatio":
				Configuration.devRatio = Double.parseDouble(kv[1]);
				break;
			case "semevalTestFile":
				Configuration.semevalTestFile = kv[1].trim();
				break;
			case "testInputFilePath":
				Configuration.testInputFilePath = kv[1].trim();
				break;
			case "testFolder":
				Configuration.testFolder = kv[1].trim();
				break;
			case "testRatio":
				Configuration.testRatio = Double.parseDouble(kv[1]);
				break;
			case "isLive":
				String isL = kv[1].trim();
				if(isL.equals("TRUE")) {
					Configuration.isLive = true;
				} else {
					Configuration.isLive = false;
				}
				break;
			default:
				break;
			}
		}
		br.close();
	}

}
