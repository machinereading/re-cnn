/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import com.twitter.penguin.korean.KoreanTokenJava;
import com.twitter.penguin.korean.TwitterKoreanProcessorJava;

import edu.kaist.mrlab.nn.pcnn.utilities.KoreanAnalyzer;
import scala.collection.Seq;

/**
 * Created by kepricon on 16. 10. 20. KoreanTokenizer using KoreanTwitterText
 * (https://github.com/twitter/twitter-korean-text) <br>
 * Modified by sangha on 17. 08. 07.
 */
public class KoreanTokenizer implements Tokenizer {
	private Iterator<String> tokenIter;
	private List<String> tokenList;

	private TokenPreProcess preProcess;

	public KoreanTokenizer(boolean flag, String toTokenize) {

		tokenList = new ArrayList<String>();

		if (flag) {

			// Tokenize with POS tag
			Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava
					.tokenize(toTokenize);
			Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava
					.stem(tokens);
			Iterator<KoreanTokenJava> iter = TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(stemmed).iterator();

			boolean isEntity = false;
			while (iter.hasNext()) {
				KoreanTokenJava ktj = iter.next();
				if (ktj.getText().equals("<<")) {
					isEntity = true;
					continue;
				} else if (ktj.getText().equals(">>")) {
					isEntity = false;
					tokenList.add("Entity");
					continue;
				}
				if (isEntity) {
				} else {
					tokenList.add(ktj.getPos().toString());
				}

			}
			tokenIter = tokenList.iterator();
		} else {

			// toTokenize = toTokenize.replace(" << ", "");
			// toTokenize = toTokenize.replace(" >> ", "");

			StringTokenizer st = new StringTokenizer(toTokenize, " ");
			boolean isEntity = false;
			String entity = "";
			while (st.hasMoreTokens()) {
				String ktj = st.nextToken();
				if (ktj.equals("<<")) {
					isEntity = true;
					continue;
				} else if (ktj.equals(">>")) {
					isEntity = false;
					tokenList.add(entity);
					entity = "";
					continue;
				}
				if (isEntity) {
					entity += ktj;
				} else {
					tokenList.add(ktj);
				}

			}
			tokenIter = tokenList.iterator();

		}

	}

	public KoreanTokenizer(String toTokenize) {

		// Tokenize with POS tag
		Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava
				.tokenize(toTokenize);
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
				tokenList.remove(tokenList.size() - 1);
				entity = tempToken.getText() + "/" + tempToken.getPos() + "/" + ktj.getText();
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
	}

	@Override
	public boolean hasMoreTokens() {
		return tokenIter.hasNext();
	}

	@Override
	public int countTokens() {
		return tokenList.size();
	}

	@Override
	public String nextToken() {
		if (hasMoreTokens() == false) {
			throw new NoSuchElementException();
		}
		return this.preProcess != null ? this.preProcess.preProcess(tokenIter.next()) : tokenIter.next();
	}

	@Override
	public List<String> getTokens() {
		return tokenList;
	}

	@Override
	public void setTokenPreProcessor(TokenPreProcess tokenPreProcess) {
		this.preProcess = tokenPreProcess;
	}
}
