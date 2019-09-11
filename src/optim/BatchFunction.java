/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package optim;

import tensor.DoubleTensor;

/**
 * Allows a to evaluate a function that uses a training parameter matrix X,
 * and a result parameter matrix y, to be trained by batches. LambdaAdjus is used
 * to adjust the regularization parameter, since it must be scaled according
 * to the batch size.
 * @author diego_paez
 */
public interface BatchFunction {
    public double value(double[] value,DoubleTensor XBatch, DoubleTensor yBatch,
            DoubleTensor yWeightsBatch, double lambdaAdjus);
}
