package edu.kaist.mrlab.nn.pcnn.iterator;

import java.util.List;

import org.deeplearning4j.berkeley.Pair;

/**
 * 
 * @author sangha
 *
 */

public interface LabeledSentencePositionProvider {
	/**
     * Are there more sentences/documents available?
     */
    boolean hasNext();

    /**
     *
     * @return Pair: sentence/document text and label
     */
    Pair<String, String> nextSentence();
    
    /**
    *
    * @return String array: sentence/document text, sbjPosition, objPosition and label
    */
    String[] nextSentenceWithPosition();

    /**
     * Reset the iterator - including shuffling the order, if necessary/appropriate
     */
    void reset();

    /**
     * Return the total number of sentences, or -1 if not available
     */
    int totalNumSentences();

    /**
     * Return the list of labels - this also defines the class/integer label assignment order
     */
    List<String> allLabels();

    /**
     * Equivalent to allLabels().size()
     */
    int numLabelClasses();
}
