package edu.kaist.mrlab.nn.pcnn.prepro;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import com.twitter.penguin.korean.KoreanTokenJava;
import com.twitter.penguin.korean.TwitterKoreanProcessorJava;

import akka.japi.Pair;
import edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizerfactory.KoreanTokenizerFactory;
import scala.collection.Seq;

public class ExtractorPreprocessor {

	BufferedWriter bw;

	public List<String> singleTest(String input) throws IOException {

		List<String> testList = new ArrayList<String>();

		String sentence;

		List<Pair<String, String>> entityPairs = getEntityPairs(input);

		for (int i = 0; i < entityPairs.size(); i++) {
			Pair<String, String> entityPair = entityPairs.get(i);
			String e1 = entityPair.first();
			String e2 = entityPair.second();

			if (e1.equals(e2) || e1.length() == 0 || e2.length() == 0) {
				continue;
			}

			sentence = input.replace("<< " + e1 + " >>", "<< _sbj_ >>");
			sentence = sentence.replace("<< " + e2 + " >>", "<< _obj_ >>");

			sentence = sentence.replaceAll("<< ", "");
			sentence = sentence.replaceAll(" >>", "");

			sentence = sentence.replaceAll("_sbj_", "<< _sbj_ >>");
			sentence = sentence.replaceAll("_obj_", "<< _obj_ >>");

			String result = e1 + "\t" + e2 + "\t" + "unknown" + "\t" + sentence;
			testList.add(singlePrepro(result, input));

		}

		return testList;
	}

	public void test(String testFile) throws IOException {

		BufferedReader br = Files.newBufferedReader(Paths.get(testFile));
		bw = Files.newBufferedWriter(Paths.get("data/test/ko_test_parsed.txt"));

		String input = null;
		String sentence = null;
		while ((input = br.readLine()) != null) {

			List<Pair<String, String>> entityPairs = getEntityPairs(input);

			for (int i = 0; i < entityPairs.size(); i++) {
				Pair<String, String> entityPair = entityPairs.get(i);
				String e1 = entityPair.first();
				String e2 = entityPair.second();

				if (e1.equals(e2) || e1.length() == 0 || e2.length() == 0) {
					continue;
				}

				sentence = input.replace("<< " + e1 + " >>", "<< _sbj_ >>");
				sentence = sentence.replace("<< " + e2 + " >>", "<< _obj_ >>");

				sentence = sentence.replaceAll("<< ", "");
				sentence = sentence.replaceAll(" >>", "");

				sentence = sentence.replaceAll("_sbj_", "<< _sbj_ >>");
				sentence = sentence.replaceAll("_obj_", "<< _obj_ >>");

				String result = e1 + "\t" + e2 + "\t" + "unknown" + "\t" + sentence;
				prepro(result, input);

			}

		}
		br.close();
		bw.close();
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

	public String singlePrepro(String sentence, String input) throws IOException {

		TokenizerFactory t = new KoreanTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		Iterator<String> tokenIter;
		List<String> tokenList;

		StringTokenizer st = new StringTokenizer(sentence, "\t");
		String sbj = st.nextToken();
		String obj = st.nextToken();
		st.nextToken(); // pred
		String stc = st.nextToken();

		// Tokenize with POS tag
		Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava
				.tokenize(stc);
		Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava
				.stem(tokens);
		tokenList = new ArrayList<String>();
		Iterator<KoreanTokenJava> iter = TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(stemmed).iterator();

		boolean isEntity = false;
		boolean isWSD = false;
		String entity = "";
		KoreanTokenJava tempToken = null;
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

			if (ktj.getText().equals("-@-")) {
				isWSD = true;
				continue;
			}

			if (isWSD) {
				entity = tempToken.getText() + "/" + tempToken.getPos() + "/" + ktj.getText();
				tokenList.remove(tokenList.size() - 1);
				tokenList.add(entity);
				entity = "";
				isWSD = false;
				continue;
			}

			if (isEntity) {
				entity += ktj.getText();
			} else {
				tokenList.add(ktj.getText() + "/" + ktj.getPos());

			}
			tempToken = ktj;

		}
		tokenIter = tokenList.iterator();

		int sbjPos = -1;
		int objPos = -1;

		t.getTokenPreProcessor();

		int count = -1;

		String reStc = "";

		while (tokenIter.hasNext()) {
			String temp = tokenIter.next();
			count++;

			if (temp.equals("_sbj_/Entity")) {
				sbjPos = count;
				temp = temp.replace("_sbj_", sbj);
			}

			if (temp.equals("_obj_/Entity")) {
				objPos = count;
				temp = temp.replace("_obj_", obj);
			}

			reStc += temp + " ";
		}

		if (sbjPos != -1 && objPos != -1) {
			return reStc + "\t" + sbjPos + "\t" + objPos + "\t" + "unknown";
		}
		return null;
	}

	public void prepro(String sentence, String input) throws IOException {

		TokenizerFactory t = new KoreanTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		Iterator<String> tokenIter;
		List<String> tokenList;

		StringTokenizer st = new StringTokenizer(sentence, "\t");
		String sbj = st.nextToken();
		String obj = st.nextToken();
		st.nextToken(); // pred
		String stc = st.nextToken();

		// Tokenize with POS tag
		Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava
				.tokenize(stc);
		Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava
				.stem(tokens);
		tokenList = new ArrayList<String>();
		Iterator<KoreanTokenJava> iter = TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(stemmed).iterator();

		boolean isEntity = false;
		boolean isWSD = false;
		String entity = "";
		KoreanTokenJava tempToken = null;
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

			if (ktj.getText().equals("-@-")) {
				isWSD = true;
				continue;
			}

			if (isWSD) {
				entity = tempToken.getText() + "/" + tempToken.getPos() + "/" + ktj.getText();
				tokenList.remove(tokenList.size() - 1);
				tokenList.add(entity);
				entity = "";
				isWSD = false;
				continue;
			}

			if (isEntity) {
				entity += ktj.getText();
			} else {
				tokenList.add(ktj.getText() + "/" + ktj.getPos());

			}
			tempToken = ktj;

		}
		tokenIter = tokenList.iterator();

		int sbjPos = -1;
		int objPos = -1;

		t.getTokenPreProcessor();

		int count = -1;

		String reStc = "";

		while (tokenIter.hasNext()) {
			String temp = tokenIter.next();
			count++;

			if (temp.equals("_sbj_/Entity")) {
				sbjPos = count;
				temp = temp.replace("_sbj_", sbj);
			}

			if (temp.equals("_obj_/Entity")) {
				objPos = count;
				temp = temp.replace("_obj_", obj);
			}

			reStc += temp + " ";
		}

		if (sbjPos != -1 && objPos != -1) {
			bw.write(reStc + "\t" + sbjPos + "\t" + objPos + "\t" + "unknown" + "\t" + input + "\n");
			// Testor.originalSentenceStore.addSentence(input);
		}
	}

	public static void main(String[] ar) throws Exception {

		ExtractorPreprocessor ep = new ExtractorPreprocessor();
		ep.test("data/test/ko_test.txt");
	}
}
