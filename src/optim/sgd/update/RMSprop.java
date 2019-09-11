package optim.sgd.update;

import java.util.Arrays;
import optim.BatchFunction;
import optim.BatchGradient;
import org.jblas.DoubleMatrix;
import tensor.DoubleTensor;

/**
 * RMSprop is a very effective, but currently unpublished adaptive learning rate
 * method. Amusingly, everyone who uses this method in their work currently
 * cites slide 29 of Lecture 6 of Geoff Hinton’s Coursera class.
 *
 * Here, decay_rate is a hyperparameter and typical values are [0.9, 0.99, 0.999].
 * Notice that the x+= update is identical to Adagrad, but the cache variable is
 * a “leaky”. Hence, RMSProp still modulates the learning rate of each weight
 * based on the magnitudes of its gradients, which has a beneficial equalizing
 * effect, but unlike Adagrad the updates do not get monotonically smaller.
 * @author diego_paez
 */
public class RMSprop extends SGDUpdate{

    public double[] cache;
    public double decay_rate;
    private final static double eps = 1E-8;

    public RMSprop(double stepSize, double decay_rate, int length) {
        this.stepSize = stepSize;
        this.decay_rate = decay_rate;
        cache = new double[length];
        Arrays.fill(cache, 0.0);
    }

    @Override
    public double update(double[] x, BatchFunction fun, BatchGradient grad,
            DoubleTensor sX, DoubleTensor sy, DoubleTensor st, double lambdaAdjus) {
        double f = fun.value(x,sX,sy,st,lambdaAdjus);
        double[] g = grad.value(x,sX,sy,st,lambdaAdjus);
        for(int i = 0; i < x.length; i++){
            cache[i] = decay_rate * cache[i] + (1 - decay_rate) * g[i] * g[i];
            x[i] += - stepSize * g[i] / (Math.sqrt(cache[i]) + eps);
        }
        return f;
    }
}