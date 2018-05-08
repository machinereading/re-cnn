package edu.kaist.mrlab.nlp.pos;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import com.twitter.penguin.korean.KoreanTokenJava;
import com.twitter.penguin.korean.TwitterKoreanProcessorJava;

import edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizerfactory.KoreanTokenizerFactory;
import scala.collection.Seq;

public class TwitterAnalyzer {

	public static void main(String[] ar) throws Exception {
		
		TokenizerFactory t = new KoreanTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		
		Iterator<String> tokenIter;
		List<String> tokenList;
		
		BufferedReader br = Files.newBufferedReader(Paths.get("data/corpus/sample.txt"));
		BufferedWriter bw = Files.newBufferedWriter(Paths.get("data/corpus/sample-parsed.txt"));
		
		String input = null;
		while((input = br.readLine()) != null){
			// Tokenize with POS tag
			
			Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava
					.tokenize(input);
//			Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava
//					.stem(tokens);
			tokenList = new ArrayList<String>();
			Iterator<KoreanTokenJava> iter = TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(tokens)
					.iterator();
			
			boolean isEntity = false;
			String entity = "";
			while (iter.hasNext()) {
				KoreanTokenJava ktj = iter.next();
				if (ktj.getText().equals("<<")) {
					isEntity = true;
					continue;
				} else if (ktj.getText().equals(">>")) {
					isEntity = false;
					tokenList.add(entity + "/Entity");
					entity = "";
					continue;
				}
				if (isEntity) {
					entity += ktj.getText();
				} else {
					tokenList.add(ktj.getText() + "/" + ktj.getPos());
				}

			}
			tokenIter = tokenList.iterator();
			
			String reStc = "";

			while (tokenIter.hasNext()) {
				String temp = tokenIter.next();
				reStc += temp + " ";
			}
			
			bw.write(reStc + "\n");
			
		}
		
		bw.close();
		System.out.println("Done!");
		
	}
}
