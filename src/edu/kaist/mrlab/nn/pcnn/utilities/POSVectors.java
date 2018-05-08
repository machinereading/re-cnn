package edu.kaist.mrlab.nn.pcnn.utilities;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 
 * @author sangha
 *
 */
public class POSVectors {
	
	int POSVectorSize = 5;

	Map<String, double[]> posEmbeddings = new HashMap<>();
	
	public void init() {
		try{
			BufferedReader br = Files.newBufferedReader(Paths.get("data/embedding/pos_num"));
			String input = null;
			String pos = null;
			while((input = br.readLine()) != null){
				
				StringTokenizer st = new StringTokenizer(input, "\t");
				pos = st.nextToken();
				double[] posEmbed = new double[5];
				int i = 0;
				while(st.hasMoreTokens()){
					posEmbed[i] = Double.parseDouble(st.nextToken());
					i++;
				}
				
				posEmbeddings.put(pos, posEmbed);
			}
		} catch (Exception e){
			e.printStackTrace();
		}
		
	}
	
	public POSVectors(){
		this.init();
	}

	public INDArray getPOSVectors(String pos){
		INDArray result = Nd4j.create(posEmbeddings.get(pos));
		return result;
		
	}
	
	public int getPOVectorSize(){
		return this.POSVectorSize;
	}
	
}
