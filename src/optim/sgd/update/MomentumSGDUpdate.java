package optim.sgd.update;

import java.util.Arrays;
import optim.BatchFunction;
import optim.BatchGradient;
import tensor.DoubleTensor;

/**
 * With Momentum update, the parameter vector will build up velocity in any
 * direction that has consistent gradient.
 * @author diego_paez
 */
public class MomentumSGDUpdate extends SGDUpdate{

    public double momentum;
    public double[] v;

    public MomentumSGDUpdate(double stepSize, double momentum, int length) {
        this.stepSize = stepSize;
        this.momentum = momentum;
        v = new double[length];
        Arrays.fill(v, 0.0);
    }

    @Override
    public double update(double[] x, BatchFunction fun, BatchGradient grad,
            DoubleTensor sX, DoubleTensor sy, DoubleTensor st, double lambdaAdjus) {
        double f = fun.value(x,sX,sy,st,lambdaAdjus);
        double[] g = grad.value(x,sX,sy,st,lambdaAdjus);
        for(int i = 0; i < x.length; i++){
            v[i] = momentum * v[i] - stepSize * g[i];
            x[i] += v[i];
        }
        return f;
    }

}
