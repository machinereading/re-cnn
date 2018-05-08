package edu.kaist.mrlab.nn.pcnn.conf.layers;

import java.util.Collection;
import java.util.Map;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 
 * @author sangha
 *
 */

@SuppressWarnings("serial")
public class PiecewiseMaxPoolingLayer extends Layer {

	private PoolingType poolingType;
	private int[] poolingDimensions;
	private int pnorm;
	private boolean collapseDimensions;

	private PiecewiseMaxPoolingLayer(Builder builder) {
		this.poolingType = builder.poolingType;
		this.poolingDimensions = builder.poolingDimensions;
		this.collapseDimensions = builder.collapseDimensions;
		this.pnorm = builder.pnorm;
		this.layerName = builder.layerName;
	}

	@Override
	public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
			Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView,
			boolean initializeParams) {
		edu.kaist.mrlab.nn.pcnn.layers.pooling.PiecewiseMaxPoolingLayer ret = new edu.kaist.mrlab.nn.pcnn.layers.pooling.PiecewiseMaxPoolingLayer(
				conf);
		ret.setListeners(iterationListeners);
		ret.setIndex(layerIndex);
		ret.setParamsViewArray(layerParamsView);
		Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
		ret.setParamTable(paramTable);
		ret.setConf(conf);
		return ret;
	}

	@Override
	public ParamInitializer initializer() {
		return EmptyParamInitializer.getInstance();
	}

	@Override
	public InputType getOutputType(int layerIndex, InputType inputType) {

		switch (inputType.getType()) {
		case FF:
			throw new UnsupportedOperationException(
					"Global max pooling cannot be applied to feed-forward input type. Got input type = " + inputType);
		case RNN:
			InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent) inputType;
			if (collapseDimensions) {
				// Return 2d (feed-forward) activations
				return InputType.feedForward(recurrent.getSize());
			} else {
				// Return 3d activations, with shape [minibatch, timeStepSize,
				// 1]
				return recurrent;
			}
		case CNN:
			InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional) inputType;
			if (collapseDimensions) {
				return InputType.feedForward(conv.getDepth());
			} else {
				return InputType.convolutional(1, 1, conv.getDepth());
			}
		case CNNFlat:
			InputType.InputTypeConvolutionalFlat convFlat = (InputType.InputTypeConvolutionalFlat) inputType;
			if (collapseDimensions) {
				return InputType.feedForward(convFlat.getDepth());
			} else {
				return InputType.convolutional(1, 1, convFlat.getDepth());
			}
		default:
			throw new UnsupportedOperationException("Unknown or not supported input type: " + inputType);
		}
	}

	@Override
	public void setNIn(InputType inputType, boolean override) {
		// Not applicable
	}

	@Override
	public InputPreProcessor getPreProcessorForInputType(InputType inputType) {

		switch (inputType.getType()) {
		case FF:
			throw new UnsupportedOperationException(
					"Global max pooling cannot be applied to feed-forward input type. Got input type = " + inputType);
		case RNN:
		case CNN:
			// No preprocessor required
			return null;
		case CNNFlat:
			InputType.InputTypeConvolutionalFlat cFlat = (InputType.InputTypeConvolutionalFlat) inputType;
			return new FeedForwardToCnnPreProcessor(cFlat.getHeight(), cFlat.getWidth(), cFlat.getDepth());
		}

		return null;
	}

	@Override
	public double getL1ByParam(String paramName) {
		// Not applicable
		return 0;
	}

	@Override
	public double getL2ByParam(String paramName) {
		// Not applicable
		return 0;
	}

	@Override
	public double getLearningRateByParam(String paramName) {
		// Not applicable
		return 0;
	}

	public PoolingType getPoolingType() {
		return poolingType;
	}

	public void setPoolingType(PoolingType poolingType) {
		this.poolingType = poolingType;
	}

	public int[] getPoolingDimensions() {
		return poolingDimensions;
	}

	public void setPoolingDimensions(int[] poolingDimensions) {
		this.poolingDimensions = poolingDimensions;
	}

	public int getPnorm() {
		return pnorm;
	}

	public void setPnorm(int pnorm) {
		this.pnorm = pnorm;
	}

	public boolean isCollapseDimensions() {
		return collapseDimensions;
	}

	public void setCollapseDimensions(boolean collapseDimensions) {
		this.collapseDimensions = collapseDimensions;
	}

	public static class Builder extends Layer.Builder<Builder> {

		private PoolingType poolingType = PoolingType.MAX;
		private int[] poolingDimensions;
		private int pnorm = 2;
		private boolean collapseDimensions = true;
		private String layerName = "piecewisePool";

		public Builder() {

		}

		public Builder(PoolingType poolingType) {
			this.poolingType = poolingType;
		}

		/**
		 * Pooling dimensions. Note: most of the time, this doesn't need to be
		 * set, and the defaults can be used. Default for RNN data: pooling
		 * dimension 2 (time). Default for CNN data: pooling dimensions 2,3
		 * (height and width)
		 *
		 * @param poolingDimensions
		 *            Pooling dimensions to use
		 */
		public Builder poolingDimensions(int... poolingDimensions) {
			this.poolingDimensions = poolingDimensions;
			return this;
		}

		/**
		 * @param poolingType
		 *            Pooling type for global pooling
		 */
		public Builder poolingType(PoolingType poolingType) {
			this.poolingType = poolingType;
			return this;
		}

		/**
		 * Whether to collapse dimensions when pooling or not. Usually you *do*
		 * want to do this. Default: true. If true:<br>
		 * - 3d (time series) input with shape [minibatchSize, vectorSize,
		 * timeSeriesLength] -> 2d output [minibatchSize, vectorSize]<br>
		 * - 4d (CNN) input with shape [minibatchSize, depth, height, width] ->
		 * 2d output [minibatchSize, depth]<br>
		 *
		 * If false:<br>
		 * - 3d (time series) input with shape [minibatchSize, vectorSize,
		 * timeSeriesLength] -> 3d output [minibatchSize, vectorSize, 1]<br>
		 * - 4d (CNN) input with shape [minibatchSize, depth, height, width] ->
		 * 2d output [minibatchSize, depth, 1, 1]<br>
		 *
		 * @param collapseDimensions
		 *            Whether to collapse the dimensions or not
		 */
		public Builder collapseDimensions(boolean collapseDimensions) {
			this.collapseDimensions = collapseDimensions;
			return this;
		}

		/**
		 * P-norm constant. Only used if using {@link PoolingType#PNORM} for the
		 * pooling type
		 *
		 * @param pnorm
		 *            P-norm constant
		 */
		public Builder pnorm(int pnorm) {
			if (pnorm <= 0)
				throw new IllegalArgumentException("Invalid input: p-norm value must be greater than 0. Got: " + pnorm);
			this.pnorm = pnorm;
			return this;
		}

		@Override
		@SuppressWarnings("unchecked")
		public PiecewiseMaxPoolingLayer build() {
			// TODO Auto-generated method stub
			return new PiecewiseMaxPoolingLayer(this);
		}
	}
}
