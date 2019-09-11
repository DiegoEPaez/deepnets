/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package optim;

import tensor.DoubleTensor;

/**
 * Allows a to evaluate a vector function (usually a gradient) that uses a training
 * parameter matrix X, and a result parameter matrix y, to be trained by batches.
 * @author diego_paez
 */
public interface BatchGradient {
    /**
     *
     * @param arg0
     * @param XBatch
     * @param yBatch
     * @param trainWeights Used to give a specific weight to train data
     * @param lambdaAdjus Used when running batches with lambda penalty, lambda
     *  must be adjusted by batchSize / totalData
     * @return
     */
    public double[] value(double[] value,DoubleTensor XBatch, DoubleTensor yBatch,
            DoubleTensor yWeightsBatch, double lambdaAdjus);
}
