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

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

/**
 * 
 * @author sangha
 *
 */
public class EnglishTokenizer implements Tokenizer {
	private Iterator<String> tokenIter;
	private List<String> tokenList;

	private TokenPreProcess preProcess;
	
	public EnglishTokenizer(String toTokenize){
		
		tokenList = new ArrayList<String>();
		
		boolean isEntity = false;
		String entity = "";
		
		String[] tokenArr = toTokenize.split(" ");
		for (int i = 0; i < tokenArr.length; i++) {

			String word = tokenArr[i];
			
			if (word.equals("-LCB-")) {
				isEntity = true;
				continue;
			} else if (word.equals("-RCB-")) {
				isEntity = false;
				tokenList.add(entity);
				entity = "";
				continue;
			}

			if (isEntity) {
				entity += word;
			} else {
				tokenList.add(word);
			}
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
