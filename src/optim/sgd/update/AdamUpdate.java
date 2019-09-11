package optim.sgd.update;

import java.util.Arrays;
import optim.BatchFunction;
import optim.BatchGradient;
import tensor.DoubleTensor;

/**
 * Notice that the update looks exactly as RMSProp update, except the “smooth”
 * version of the gradient m is used instead of the raw (and perhaps noisy)
 * gradient vector dx.
 *
 * Recommended values in the paper are eps = 1e-8, beta1 = 0.9,
 * beta2 = 0.999. In practice Adam is currently recommended as the default algorithm
 * to use, and often works slightly better than RMSProp.
 *
 * See "ADAM: A Method For Stochastic Optimization"
 * @author diego_paez
 */
public class AdamUpdate extends SGDUpdate{

    public double[] cache1;
    public double[] cache2;
    public double beta1;
    public double beta2;
    public int iter;
    private final static double eps = 1E-8;

    public AdamUpdate(double stepSize, double beta1, double beta2, int length) {
        this.stepSize = stepSize;
        this.beta1 = beta1;
        this.beta2 = beta2;
        cache1 = new double[length];
        cache2 = new double[length];
        Arrays.fill(cache1, 0.0);
        Arrays.fill(cache2, 0.0);
        iter = 0;
    }

    @Override
    public double update(double[] x, BatchFunction fun, BatchGradient grad,
            DoubleTensor sX, DoubleTensor sy, DoubleTensor st, double lambdaAdjus) {
        double f = fun.value(x,sX,sy,st,lambdaAdjus);
        double[] g = grad.value(x,sX,sy,st,lambdaAdjus);
        double biasCorrection1, biasCorrection2;
        iter++;

        for(int i = 0; i < x.length; i++){
            cache1[i] = beta1 * cache1[i] + (1 - beta1) * g[i];
            cache2[i] = beta2 * cache2[i] + (1 - beta2) * g[i] * g[i];
            biasCorrection1 = cache1[i] / (1 - Math.pow(beta1,iter));
            biasCorrection2 = cache2[i] / (1 - Math.pow(beta2,iter));
            x[i] += - stepSize * biasCorrection1 / (Math.sqrt(biasCorrection2) + eps);
        }
        return f;
    }
}