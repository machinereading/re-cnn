package edu.kaist.mrlab.nn.pcnn.prepro;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import com.twitter.penguin.korean.KoreanTokenJava;
import com.twitter.penguin.korean.TwitterKoreanProcessorJava;

import edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizerfactory.KoreanTokenizerFactory;
import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;
import edu.kaist.mrlab.nn.pcnn.utilities.OntoProperty;
import scala.collection.Seq;

/**
 * 
 * @author sangha
 *
 */
public class SentencePreprocessorWithTokenizer {

	private Path inputPath;
	private Path outputPath;

	private Iterator<String> tokenIter;
	private List<String> tokenList;

	public void getWords(String property) throws Exception {
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		BufferedWriter bw = Files.newBufferedWriter(outputPath);
		BufferedReader br = Files.newBufferedReader(inputPath);

		String input = null;
		while ((input = br.readLine()) != null) {

			// System.out.println(input);

			StringTokenizer st = new StringTokenizer(input, "\t");
			String sbj = st.nextToken();
			String obj = st.nextToken();
			String pred = st.nextToken();
			String stc = st.nextToken();

			if (pred.equals(property)) {
				tokenList = new ArrayList<String>();

				boolean isEntity = false;
				String entity = "";

				StringTokenizer st2 = new StringTokenizer(stc, " ");

				while (st2.hasMoreTokens()) {
					String token = st2.nextToken();
					if (token.equals("<<")) {
						isEntity = true;
						continue;
					} else if (token.equals(">>")) {
						isEntity = false;

						tokenList.add(entity.trim().replace(" ", "_"));
						entity = "";
						continue;
					}

					if (isEntity) {
						entity += token;
					} else {
						tokenList.add(token);
					}
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

					if (temp.equals("_sbj_")) {
						sbjPos = count;
						temp = temp.replace("_sbj_", sbj);
					}

					if (temp.equals("_obj_")) {
						objPos = count;
						temp = temp.replace("_obj_", obj);
					}

					reStc += temp + " ";
				}

				if (sbjPos < 0 || objPos < 0) {
					System.out.println(reStc + "\t" + sbjPos + "\t" + objPos + "\t" + property + "\n");
				} else {
					bw.write(reStc + "\t" + sbjPos + "\t" + objPos + "\t" + property + "\t" + stc + "\n");
				}

			}
		}
		bw.close();

	}

	public void getStems(boolean withPOS, boolean withWSD, String property) throws Exception {

		TokenizerFactory t = new KoreanTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		BufferedWriter bw = Files.newBufferedWriter(outputPath);
		BufferedReader br = Files.newBufferedReader(inputPath);

		String input = null;
		while ((input = br.readLine()) != null) {

			// System.out.println(input);

			StringTokenizer st = new StringTokenizer(input, "\t");
			String sbj = st.nextToken();
			String obj = st.nextToken();
			String pred = st.nextToken();
			String stc = st.nextToken();

			if (pred.equals(property)) {

				// Tokenize with POS tag
				Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava
						.tokenize(stc);
				Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava
						.stem(tokens);
				tokenList = new ArrayList<String>();
				Iterator<KoreanTokenJava> iter = TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(stemmed)
						.iterator();

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

						if (withPOS) {
							tokenList.add(entity + "/Entity");
						} else {
							tokenList.add(entity.trim().replace(" ", "_"));
						}

						entity = "";
						continue;
					}

					if (withWSD) {
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
					}

					if (isEntity) {
						entity += ktj.getText();
					} else {
						if (withPOS) {
							tokenList.add(ktj.getText() + "/" + ktj.getPos());
						} else {
							tokenList.add(ktj.getText());
						}

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

					if (withPOS) {
						if (temp.equals("_sbj_/Entity")) {
							sbjPos = count;
							temp = temp.replace("_sbj_", sbj);
						}

						if (temp.equals("_obj_/Entity")) {
							objPos = count;
							temp = temp.replace("_obj_", obj);
						}
					} else {
						if (temp.equals("_sbj_")) {
							sbjPos = count;
							temp = temp.replace("_sbj_", sbj);
						}

						if (temp.equals("_obj_")) {
							objPos = count;
							temp = temp.replace("_obj_", obj);
						}
					}

					reStc += temp + " ";
				}

				if (sbjPos < 0 || objPos < 0) {
					System.out.println(reStc + "\t" + sbjPos + "\t" + objPos + "\t" + property + "\n");
				} else {
					bw.write(reStc + "\t" + sbjPos + "\t" + objPos + "\t" + property + "\t" + stc + "\n");
				}

				// bw.write(reStc + "\n");

			}

		}
		bw.close();

	}

	public SentencePreprocessorWithTokenizer setInputPath(Path inputPath) {
		this.inputPath = inputPath;
		return this;
	}

	public SentencePreprocessorWithTokenizer setOutputPath(Path outputPath) {
		this.outputPath = outputPath;
		return this;
	}

	public void run(String inputPath, String outputPath, boolean isPOS, boolean isWSD) throws Exception {
		System.out.println("Parsing POS tag to DS sentence...");

		for (OntoProperty p : OntoProperty.values()) {
			String property = p.toString();

			if (property.equals("classP")) {
				property = "class";
			}

			System.out.print(property + "...");

			File f = new File(outputPath);
			if (!f.exists()) {
				f.mkdirs();
			}

			setInputPath(Paths.get(inputPath)).setOutputPath(Paths.get(outputPath, property)).getStems(isPOS, isWSD,
					property);

			System.out.println("Done!");
		}
	}

	public static void main(String[] ar) throws Exception {

		SentencePreprocessorWithTokenizer spwt = new SentencePreprocessorWithTokenizer();
		spwt.run("data/ds/iter/kowiki-20170701-kbox_initial-wikilink-iter4-except-gold",
				"data/ds/iter/total_" + GlobalVariables.option + "/", true, false);

	}
}
