package edu.kaist.mrlab.nn.pcnn;

import java.nio.file.Paths;

import edu.kaist.mrlab.nn.pcnn.pipeline.Extraction;
import edu.kaist.mrlab.nn.pcnn.pipeline.Learning;

public class Main {
	public static void main(String[] ar) throws Exception {
		
		Configuration.load(Paths.get("PCNN.conf"));
		
		boolean training = false;
		boolean testing = false;
		
		for (int i = 0; i < ar.length; i++) {
			if(ar[i].equals("--training")) training = true;
			else if(ar[i].equals("--testing")) testing = true;
		}
		
		if(training) Learning.learn();
		if(testing) Extraction.test(); 
		
	}
}
