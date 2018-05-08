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
public class LabelVectors {
	
	int labelVectorSize = 100;

	Map<String, double[]> labelEmbeddings = new HashMap<>();
	
	public void init() {
		try{
			BufferedReader br = Files.newBufferedReader(Paths.get("data/embedding/dep_label"));
			String input = null;
			String label = null;
			while((input = br.readLine()) != null){
				
				StringTokenizer st = new StringTokenizer(input, "\t");
				label = st.nextToken();
				double[] labelEmbed = new double[100];
				int i = 0;
				while(st.hasMoreTokens()){
					labelEmbed[i] = Double.parseDouble(st.nextToken());
					i++;
				}
				
				labelEmbeddings.put(label, labelEmbed);
			}
		} catch (Exception e){
			e.printStackTrace();
		}
		
	}
	
	public LabelVectors(){
		this.init();
	}

	public INDArray getLabelVectors(String label){
		INDArray result = Nd4j.create(labelEmbeddings.get(label));
		return result;
		
	}
	
	public int getLabelVectorSize(){
		return this.labelVectorSize;
	}
	
}
