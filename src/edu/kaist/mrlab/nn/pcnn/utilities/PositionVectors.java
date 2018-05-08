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
public class PositionVectors {
	
	int positionVectorSize = 5;

	Map<Integer, double[]> posEmbeddings = new HashMap<>();
	
	public void init() {
		try{
			BufferedReader br = Files.newBufferedReader(Paths.get("data/embedding/position"));
			String input = null;
			int count = 0;
			while((input = br.readLine()) != null){
				
				StringTokenizer st = new StringTokenizer(input, "\t");
				double[] posEmbed = new double[5];
				int i = 0;
				while(st.hasMoreTokens()){
					posEmbed[i] = Double.parseDouble(st.nextToken());
					i++;
				}
				
				posEmbeddings.put(count, posEmbed);
				count++;
			}
		} catch (Exception e){
			e.printStackTrace();
		}
		
	}
	
	public PositionVectors(){
		this.init();
	}

	public INDArray getPositionVectors(int position){
		INDArray result = Nd4j.create(posEmbeddings.get(position));
		return result;
		
	}
	
	public int getPositionVectorSize(){
		return this.positionVectorSize;
	}
	
}
