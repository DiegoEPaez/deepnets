package optim.sgd.update;

import java.util.Arrays;
import optim.BatchFunction;
import optim.BatchGradient;
import org.jblas.DoubleMatrix;
import tensor.DoubleTensor;

/**
 * Recommended settings: alfa = 0.002, beta1 = 0.9, beta2 = 0.999.
 *
 * See "ADAM: A Method For Stochastic Optimization"
 * @author diego_paez
 */
public class AdaMaxUpdate extends SGDUpdate{

    public double[] cache1;
    public double[] cache2;
    public double beta1;
    public double beta2;
    public int iter;

    public AdaMaxUpdate(double stepSize, double beta1, double beta2, int length) {
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
        iter++;

        for(int i = 0; i < x.length; i++){
            cache1[i] = beta1 * cache1[i] + (1 - beta1) * g[i];
            cache2[i] = Math.max(beta2 * cache2[i], Math.abs(g[i]));
            x[i] += - stepSize / (1 - Math.pow(beta1,iter)) * cache1[i] /cache2[i];
        }
        return f;
    }

}
