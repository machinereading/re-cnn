package edu.kaist.mrlab.nn.pcnn.prepro;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;

import edu.kaist.mrlab.nn.pcnn.utilities.OntoProperty;

/**
 * 
 * @author sangha
 *
 */
public class TrainTestSetSeperator {

	double testRatio = 0;

	public TrainTestSetSeperator(double testRatio) {
		this.testRatio = testRatio;
	}

	public void divide(String input, String output, String property) {

		try {

			File f = new File(output);
			if (!f.exists()) {
				f.mkdirs();
			}

			Path inputPath = Paths.get(input, property);
			Path trainOutPath = Paths.get(output, property + "_train");
			Path testOutPath = Paths.get(output, property + "_test");

			BufferedReader br = Files.newBufferedReader(inputPath);
			BufferedWriter bwTrain = Files.newBufferedWriter(trainOutPath);
			BufferedWriter bwTest = Files.newBufferedWriter(testOutPath);

			int totalCount = (int) br.lines().count();
			int testCount = (int) (totalCount * testRatio);

			br.close();

			br = Files.newBufferedReader(inputPath);
			Iterator<String> brit = br.lines().iterator();

			int count = 0;
			int tCount = 0;
			while (brit.hasNext()) {

				if (count < testCount) {
					bwTest.write(brit.next() + "\n");
					tCount++;
				} else {
					bwTrain.write(brit.next() + "\n");
				}

				count++;
			}
			
			if(tCount == 0) {
				bwTest.write("./Punctuation ./Punctuation\t0\t1\t" + property + "\t << _sbj_ >>  << _obj_ >> \n");
			}

			bwTrain.close();
			bwTest.close();
			br.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void run(String input, String output) {
		System.out.println("Divide total to train and test");

		for (OntoProperty p : OntoProperty.values()) {

			String property = p.toString();
			if (property.equals("classP")) {
				property = "class";
			}

			System.out.print(property + "...");

			divide(input, output, property);

			System.out.println("Done!");

		}
	}

	public static void main(String[] ar) {

		// String root = "data/ds/iter/";
		//
		// TrainTestSetSeperator ttss = new TrainTestSetSeperator();
		// ttss.run(root);

	}

}
