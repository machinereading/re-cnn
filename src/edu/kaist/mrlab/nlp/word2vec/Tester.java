package edu.kaist.mrlab.nlp.word2vec;

import java.util.Collection;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.kaist.mrlab.nn.pcnn.utilities.GlobalVariables;

public class Tester {

	private static Logger log = LoggerFactory.getLogger(Tester.class);

	public static void main(String[] ar) {

		long startTime = System.currentTimeMillis();
		Word2Vec vec = WordVectorSerializer
				.readWord2VecModel("data/embedding/ko_vec_100dim_1min_title_" + GlobalVariables.option);
		
//		Word2Vec vec = WordVectorSerializer
//				.readWord2VecModel("data/embedding/pos");
		
		long endTime = System.currentTimeMillis();
		System.out.println("## 소요시간(초.0f) : " + (endTime - startTime) / 1000.0f + "초");
		
//		System.out.println(vec.getWordVectorMatrix("Entity"));
//		System.out.println(vec.getWordVectorMatrix("Noun"));
//		System.out.println(vec.getWordVectorMatrix("Punctuation"));
//		System.out.println(vec.getWordVectorMatrix("Suffix"));
//		System.out.println(vec.getWordVectorMatrix("Josa"));
//		System.out.println(vec.getWordVectorMatrix("Verb"));
//		System.out.println(vec.getWordVectorMatrix("Number"));
//		System.out.println(vec.getWordVectorMatrix("Foreign"));
//		System.out.println(vec.getWordVectorMatrix("Alpha"));
//		System.out.println(vec.getWordVectorMatrix("ProperNoun"));
//		System.out.println(vec.getWordVectorMatrix("Conjunction"));
//		System.out.println(vec.getWordVectorMatrix("Adjective"));
//		System.out.println(vec.getWordVectorMatrix("Adverb"));
//		System.out.println(vec.getWordVectorMatrix("Determiner"));
//		System.out.println(vec.getWordVectorMatrix("Exclamation"));
//		System.out.println(vec.getWordVectorMatrix("VerbPrefix"));
//		System.out.println(vec.getWordVectorMatrix("NounPrefix"));
		
		System.out.println();
		
//		System.out.println(vec.getWordVectorMatrix("PreEomi"));
//		System.out.println(vec.getWordVectorMatrix("Eomi"));
//		System.out.println(vec.getWordVectorMatrix("KoreanParticle"));
//		System.out.println(vec.getWordVectorMatrix("Hashtag"));
//		System.out.println(vec.getWordVectorMatrix("ScreenName"));
//		System.out.println(vec.getWordVectorMatrix("Email"));
//		System.out.println(vec.getWordVectorMatrix("URL"));
		
		log.info("Closest Words:");
		Collection<String> lst = vec.wordsNearest("soccerclub/Alpha", 20);
		System.out.println("10 Words closest to 'soccerclub/Alpha': " + lst);

		lst = vec.wordsNearest("country/Alpha", 10);
		System.out.println("10 Words closest to 'country/Alpha': " + lst);
		
		lst = vec.wordsNearest("평창/Entity", 10);
		System.out.println("10 Words closest to '평창/Entity': " + lst);

		lst = vec.wordsNearest("강원도/Entity", 10);
		System.out.println("10 Words closest to '강원도/Entity': " + lst);

		lst = vec.wordsNearest("한국/Entity", 10);
		System.out.println("10 Words closest to '한국/Entity': " + lst);

		lst = vec.wordsNearest("미국/Entity", 10);
		System.out.println("10 Words closest to '미국/Entity': " + lst);
		
		lst = vec.wordsNearest("중국/Entity", 10);
		System.out.println("10 Words closest to '중국/Entity': " + lst);
		
		lst = vec.wordsNearest("네덜란드/Entity", 10);
		System.out.println("10 Words closest to '네덜란드/Entity': " + lst);
		
		lst = vec.wordsNearest("스위스/Entity", 10);
		System.out.println("10 Words closest to '스위스/Entity': " + lst);

	}
}
