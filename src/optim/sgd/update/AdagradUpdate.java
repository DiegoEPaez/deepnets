package optim.sgd.update;

import java.util.Arrays;
import optim.BatchFunction;
import optim.BatchGradient;
import tensor.DoubleTensor;

/**
 * A downside of Adagrad is that in case of Deep Learning, the monotonic learning
 * rate usually proves too aggressive and stops learning too early.
 * @author diego_paez
 */
public class AdagradUpdate extends SGDUpdate{

    double[] cache;
    private final static double eps = 1E-8;
    
    public AdagradUpdate(double stepSize, int length) {
        this.stepSize = stepSize;
        cache = new double[length];
        Arrays.fill(cache, 0.0);
    }

    @Override
    public double update(double[] x, BatchFunction fun, BatchGradient grad,
            DoubleTensor sX, DoubleTensor sy, DoubleTensor st, double lambdaAdjus) {
        double f = fun.value(x,sX,sy,st,lambdaAdjus);
        double[] g = grad.value(x,sX,sy,st,lambdaAdjus);
        for(int i = 0; i < x.length; i++){
            cache[i] += g[i] * g[i];
            x[i] += - stepSize * g[i] / (Math.sqrt(cache[i]) + eps);
        }
        return f;
    }
}
