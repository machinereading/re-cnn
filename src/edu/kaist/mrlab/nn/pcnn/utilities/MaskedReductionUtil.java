package edu.kaist.mrlab.nn.pcnn.utilities;

import java.util.List;

import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MaskedReductionUtil {
	private static final int[] CNN_DIM_MASK_H = new int[] { 0, 2 };
	private static final int[] CNN_DIM_MASK_W = new int[] { 0, 3 };

	public static INDArray getPiecewiseMaxIND(INDArray withInf, List<Integer> sbjPosList, List<Integer> objPosList,
			INDArray result) {
		// algorithm is so naive. It have to replaced more efficient way
		// later...

		INDArray piecewiseMaxPooled = result;
		int sbjPos = -1;
		int objPos = -1;
		INDArray log = null;

		try {

			for (int i = 0; i < withInf.size(0); i++) {

				int c = 0;
				INDArray poolingDepth = withInf.getRow(i);

				int[] shape = new int[2];
				shape[0] = 1;
				shape[1] = poolingDepth.size(0) * 3;
				INDArray oneRow = Nd4j.create(shape);

				sbjPos = sbjPosList.get(i);
				objPos = objPosList.get(i);

				int length = poolingDepth.getRow(0).rows();

				// System.out.println(sbjPos + "\t" + objPos + "\t" + length +
				// "\t" + log);

				for (int j = 0; j < poolingDepth.size(0); j++) {

					INDArray sliceOfDepth = poolingDepth.getRow(j);

					INDArray firthBunch = null;
					INDArray secondBunch = null;
					INDArray thirdBunch = null;

					log = sliceOfDepth;

					if (sbjPos < objPos) {

						// sbj obj x y z
						// sbj x obj y z
						// sbj x y z obj
						if (sbjPos == 0) {
							if (sbjPos + 1 == objPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								secondBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								if (objPos + 1 == length) {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								} else {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, length));
								}
							} else if (objPos + 1 < length) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								secondBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, objPos));
								if (objPos + 1 == length) {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								} else {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, length));
								}

							} else if (objPos + 1 == length) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								secondBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, objPos));
								thirdBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
							} else {
								
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								if(sbjPos + 1 == length){
									secondBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								} else{
									secondBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, length));
								}
								thirdBunch = null;
								
//								System.out.println("Length is not valid");
//								System.out.println(sbjPos + "\t" + objPos + "\t" + log);
//								System.exit(0);
							}
						}

						// x sbj obj y z
						// x sbj y obj z

						else if (sbjPos > 0 && objPos < length) {
							if (sbjPos + 1 == objPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, sbjPos));
								secondBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								if (objPos + 1 == length) {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								} else {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, length));
								}
							} else if (sbjPos + 1 < objPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, sbjPos));
								secondBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, objPos));
								if (objPos + 1 == length) {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								} else {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, length));
								}

							} else {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, sbjPos));
								if(sbjPos + 1 == length){
									secondBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								} else{
									secondBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, length));
								}
								
								thirdBunch = null;
//								System.out.println("Length is not valid");
//								System.out.println(sbjPos + "\t" + objPos + "\t" + log);
//								System.exit(0);
							}
						}

						// x y z sbj obj
						// x y sbj z obj

						else if (sbjPos > 0 && objPos + 1 == length) {
							if (sbjPos + 1 == objPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, sbjPos));
								secondBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								thirdBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
							} else if (sbjPos + 1 < objPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, sbjPos));
								secondBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, objPos));
								thirdBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
							} else {
								
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								if(sbjPos + 1 == length){
									secondBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								} else{
									secondBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, length));
								}
								thirdBunch = null;
								
//								System.out.println("Length is not valid");
//								System.out.println(sbjPos + "\t" + objPos + "\t" + log);
//								System.exit(0);
							}
						} else {

							int f1 = (int) (length / 3);
							int f2 = f1 * 2;

							firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, f1));
							secondBunch = sliceOfDepth.get(NDArrayIndex.interval(f1 + 1, f2));
							thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(f2, length));
						}

					} else {

						// obj sbj x y z
						// obj x sbj y z
						// obj x y z sbj

						if (objPos == 0) {
							if (objPos + 1 == sbjPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								secondBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								if (sbjPos + 1 == length) {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								} else {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, length));
								}
							} else if (sbjPos + 1 < length) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								secondBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, sbjPos));
								if (sbjPos + 1 == length) {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								} else {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, length));
								}
							} else if (sbjPos + 1 == length) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								secondBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, sbjPos));
								thirdBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
							} else {
								
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								if(objPos + 1 == length){
									secondBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								} else{
									secondBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, length));
								}
								thirdBunch = null;
								
//								System.out.println("Length is not valid");
//								System.out.println(sbjPos + "\t" + objPos + "\t" + log);
//								System.exit(0);
							}
						}

						// x obj sbj y z
						// x obj y sbj z

						else if (objPos > 0 && sbjPos < length) {
							if (objPos + 1 == sbjPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, objPos));
								secondBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								if (sbjPos + 1 == length) {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								} else {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, length));
								}

							} else if (objPos + 1 < sbjPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, objPos));
								secondBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, sbjPos));
								if (sbjPos + 1 == length) {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								} else {
									thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1, length));
								}
							} else {
								
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								if(objPos + 1 == length){
									secondBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								} else{
									secondBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, length));
								}
								thirdBunch = null;
								
//								System.out.println("Length is not valid");
//								System.out.println(sbjPos + "\t" + objPos + "\t" + log);
//								System.exit(0);
							}
						}

						// x y z obj sbj
						// x y obj z sbj

						else if (objPos > 0 && sbjPos + 1 == length) {
							if (objPos + 1 == sbjPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, objPos));
								secondBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
								thirdBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
							} else if (objPos + 1 < sbjPos) {
								firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, objPos));
								secondBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, sbjPos));
								thirdBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos), NDArrayIndex.all());
							} else {
								
								firthBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								if(objPos + 1 == length){
									secondBunch = sliceOfDepth.get(NDArrayIndex.point(objPos), NDArrayIndex.all());
								} else{
									secondBunch = sliceOfDepth.get(NDArrayIndex.interval(objPos + 1, length));
								}
								thirdBunch = null;
								
//								System.out.println("Length is not valid");
//								System.out.println(sbjPos + "\t" + objPos + "\t" + log);
//								System.exit(0);
							}
						} else {
							
							int f1 = (int) (length / 3);
							int f2 = f1 * 2;

							firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0, f1));
							secondBunch = sliceOfDepth.get(NDArrayIndex.interval(f1 + 1, f2));
							thirdBunch = sliceOfDepth.get(NDArrayIndex.interval(f2, length));

//							System.out.println("Length is not valid");
//							System.out.println(sbjPos + "\t" + objPos + "\t" + log);
//							System.exit(0);
						}
					}

					// // sbj x y z
					// if (sbjPos == 0) {
					// firthBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos),
					// NDArrayIndex.all());
					// }
					// // x sbj y z
					// else {
					// firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0,
					// sbjPos));
					// }
					//
					// // x sbj obj y z
					// if (sbjPos + 1 == objPos && objPos < sliceOfDepth.rows())
					// {
					// secondBunch =
					// sliceOfDepth.get(NDArrayIndex.point(objPos),
					// NDArrayIndex.all());
					// }
					// // x y sbj obj
					// else if (sbjPos + 1 == objPos && objPos ==
					// sliceOfDepth.rows()) {
					// secondBunch =
					// sliceOfDepth.get(NDArrayIndex.point(sbjPos),
					// NDArrayIndex.all());
					// }
					// // x sbj y obj
					// else {
					// secondBunch =
					// sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1,
					// objPos));
					// }
					//
					// if (objPos + 1 == sliceOfDepth.rows()) {
					// thirdBunch = sliceOfDepth.get(NDArrayIndex.point(objPos),
					// NDArrayIndex.all());
					// } else if (objPos + 1 > sliceOfDepth.rows()) {
					// System.out.println(sbjPos + "\t" + objPos + "\t" + log);
					// thirdBunch =
					// sliceOfDepth.get(NDArrayIndex.point(sliceOfDepth.rows() -
					// 1),
					// NDArrayIndex.all());
					// } else {
					// thirdBunch =
					// sliceOfDepth.get(NDArrayIndex.interval(objPos + 1,
					// sliceOfDepth.rows()));
					// }
					//
					// } else {
					//
					// if (objPos == 0) {
					// firthBunch = sliceOfDepth.get(NDArrayIndex.point(objPos),
					// NDArrayIndex.all());
					// } else {
					// firthBunch = sliceOfDepth.get(NDArrayIndex.interval(0,
					// objPos));
					// }
					//
					// if (objPos + 1 == sbjPos && sbjPos < sliceOfDepth.rows())
					// {
					// secondBunch =
					// sliceOfDepth.get(NDArrayIndex.point(sbjPos),
					// NDArrayIndex.all());
					// } else if (objPos + 1 == sbjPos && sbjPos ==
					// sliceOfDepth.rows()) {
					// secondBunch =
					// sliceOfDepth.get(NDArrayIndex.point(objPos),
					// NDArrayIndex.all());
					// } else {
					// secondBunch =
					// sliceOfDepth.get(NDArrayIndex.interval(objPos + 1,
					// sbjPos));
					// }
					//
					// if (sbjPos + 1 == sliceOfDepth.rows()) {
					// thirdBunch = sliceOfDepth.get(NDArrayIndex.point(sbjPos),
					// NDArrayIndex.all());
					// } else if (sbjPos + 1 > sliceOfDepth.rows()) {
					// System.out.println(sbjPos + "\t" + objPos + "\t" + log);
					// thirdBunch =
					// sliceOfDepth.get(NDArrayIndex.point(sliceOfDepth.rows() -
					// 1),
					// NDArrayIndex.all());
					// } else {
					// thirdBunch =
					// sliceOfDepth.get(NDArrayIndex.interval(sbjPos + 1,
					// sliceOfDepth.rows()));
					// }
					// }

					// if (firthBunch == null || secondBunch == null ||
					// thirdBunch == null) {
					// System.out.println(sbjPos + "\t" + objPos + "\t" + length
					// + "\t" + log);
					// }
					
					
					INDArray firstMax = firthBunch.max(0);
					INDArray secondMax = secondBunch.max(0);
					INDArray thirdMax = Nd4j.zeros(1);
					
					if(thirdBunch == null){
					} else {
						thirdMax = thirdBunch.max(0);
					}

					oneRow.putScalar(c, firstMax.getDouble(0));
					c++;
					oneRow.putScalar(c, secondMax.getDouble(0));
					c++;
					oneRow.putScalar(c, thirdMax.getDouble(0));
					c++;

				}

				piecewiseMaxPooled.putRow(i, oneRow);

			}

		} catch (IllegalArgumentException e) {

			System.out.println(sbjPos + "\t" + objPos + "\t" + log);

			e.printStackTrace();
		}
		return piecewiseMaxPooled;

	}

	// for training step
	public static INDArray piecewisePoolingConvolution(PoolingType poolingType, INDArray toReduce, INDArray mask,
			boolean alongHeight, List<Integer> sbjPos, List<Integer> objPos) {
		// [minibatch, depth, h=1, w=X] or [minibatch, depth, h=X, w=1] data
		// with a mask array of shape [minibatch, X]

		// If masking along height: broadcast dimensions are [0,2]
		// If masking along width: broadcast dimensions are [0,3]

		int[] dimensions = (alongHeight ? CNN_DIM_MASK_H : CNN_DIM_MASK_W);

		int[] resultShape = new int[2];
		resultShape[0] = toReduce.size(0);
		resultShape[1] = toReduce.size(1) * 3;

		INDArray result = Nd4j.create(resultShape);

		switch (poolingType) {
		case MAX:
			// TODO This is ugly - replace it with something better... there is
			// no difference override function for testing from a result
			// perspective.
			INDArray negInfMask = Transforms.not(mask);
			BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

			INDArray withInf = Nd4j.createUninitialized(toReduce.shape());
			Nd4j.getExecutioner().exec(new BroadcastAddOp(toReduce, negInfMask, withInf, dimensions));
			// At this point: all the masked out steps have value -inf, hence
			// can't be the output of the MAX op

			result = getPiecewiseMaxIND(withInf, sbjPos, objPos, result);

			BooleanIndexing.replaceWhere(result, 0.0, Conditions.equals(Double.NEGATIVE_INFINITY));

			// return withInf.max(2, 3);
			return result;
		default:
			throw new UnsupportedOperationException(
					"Only available MAX Type. Unknown or not supported pooling type: " + poolingType);
		}
	}

	// for testing step
	public static INDArray piecewisePoolingConvolution(PoolingType poolingType, INDArray toReduce, List<Integer> sbjPos,
			List<Integer> objPos) {

		int[] resultShape = new int[2];
		resultShape[0] = toReduce.size(0);
		resultShape[1] = toReduce.size(1) * 3;
		INDArray result = Nd4j.create(resultShape);

		switch (poolingType) {
		case MAX:
			INDArray withInf = toReduce;
			result = getPiecewiseMaxIND(withInf, sbjPos, objPos, result);
			BooleanIndexing.replaceWhere(result, 0.0, Conditions.equals(Double.NEGATIVE_INFINITY));
			return result;
		default:
			throw new UnsupportedOperationException(
					"Only available MAX Type. Unknown or not supported pooling type: " + poolingType);
		}
	}

	public static INDArray piecewisePoolingEpsilonCnn(PoolingType poolingType, INDArray input, INDArray mask,
			INDArray epsilon2d, boolean alongHeight, int pnorm) {

		// [minibatch, depth, h=1, w=X] or [minibatch, depth, h=X, w=1] data
		// with a mask array of shape [minibatch, X]

		// If masking along height: broadcast dimensions are [0,2]
		// If masking along width: broadcast dimensions are [0,3]

		int[] dimensions = (alongHeight ? CNN_DIM_MASK_H : CNN_DIM_MASK_W);

		switch (poolingType) {
		case MAX:
			// TODO This is ugly - replace it with something better... Need
			// something like a Broadcast CAS op
			INDArray negInfMask = Transforms.not(mask);
			BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

			INDArray withInf = Nd4j.createUninitialized(input.shape());
			Nd4j.getExecutioner().exec(new BroadcastAddOp(input, negInfMask, withInf, dimensions));
			// At this point: all the masked out steps have value -inf, hence
			// can't be the output of the MAX op

			INDArray isMax = Nd4j.getExecutioner().execAndReturn(new IsMax(withInf, 2, 3));

			return Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(isMax, epsilon2d, isMax, 0, 1));

		default:
			throw new UnsupportedOperationException(
					"Only available MAX Type. Unknown or not supported pooling type: " + poolingType);

		}
	}
}
