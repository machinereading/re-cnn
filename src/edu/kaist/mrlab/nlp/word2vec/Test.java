package edu.kaist.mrlab.nlp.word2vec;

import java.util.Iterator;

import com.twitter.penguin.korean.KoreanTokenJava;
import com.twitter.penguin.korean.TwitterKoreanProcessorJava;

import scala.collection.Seq;

public class Test {
	public static void main(String[] ar) {

		// Tokenize with POS tag
		Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava
				.tokenize("FC_바르셀로나(soccerclub)의 국가는 스페인(country)이다.");
		Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava
				.stem(tokens);
		Iterator<KoreanTokenJava> iter = TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(stemmed).iterator();

		while (iter.hasNext()) {
			KoreanTokenJava ktj = iter.next();
			System.out.println(ktj.getText() + ", " + ktj.getPos().toString());

		}
	}
}
