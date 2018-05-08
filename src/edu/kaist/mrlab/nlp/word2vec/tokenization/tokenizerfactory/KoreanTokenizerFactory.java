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

package edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizerfactory;

import java.io.InputStream;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import edu.kaist.mrlab.nlp.word2vec.tokenization.tokenizer.KoreanTokenizer;

/**
 * 
 * @author sangha
 *
 */
public class KoreanTokenizerFactory implements TokenizerFactory {

    private TokenPreProcess preProcess;

    public KoreanTokenizerFactory() {}

    @Override
    public Tokenizer create(String toTokenize) {
        KoreanTokenizer t = null;
		try {
			t = new KoreanTokenizer(toTokenize);
			t.setTokenPreProcessor(preProcess);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        return t;
    }

    @Override
    public Tokenizer create(InputStream inputStream) {
        throw new UnsupportedOperationException("Not supported");
        //        return null;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcess) {
        this.preProcess = tokenPreProcess;
    }

    @Override
    public TokenPreProcess getTokenPreProcessor() {
        return this.preProcess;
    }
}
