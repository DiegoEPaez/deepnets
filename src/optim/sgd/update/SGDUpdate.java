/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package optim.sgd.update;

import optim.BatchFunction;
import optim.BatchGradient;
import tensor.DoubleTensor;

/**
 *
 * @author diego_paez
 */
public abstract class SGDUpdate {

    public double stepSize;

    public abstract double update(double[] x, BatchFunction fun, BatchGradient grad,
            DoubleTensor sX, DoubleTensor sy, DoubleTensor st, double lambdaAdjus);
}
