package optim.sgd.update;

import optim.BatchFunction;
import optim.BatchGradient;
import tensor.DoubleTensor;


/**
 *
 * @author diego_paez
 */
public class PlainVanillaSGDUpdate extends SGDUpdate{

    public PlainVanillaSGDUpdate(double stepSize) {
        this.stepSize = stepSize;
    }

    @Override
    public double update(double[] x, BatchFunction fun, BatchGradient grad,
            DoubleTensor sX, DoubleTensor sy, DoubleTensor st, double lambdaAdjus) {
        double f = fun.value(x,sX,sy,st,lambdaAdjus);
        double[] g = grad.value(x,sX,sy,st,lambdaAdjus);
        for(int i = 0; i < x.length; i++){
            x[i] -= stepSize * g[i];
        }
        return f;
    }

}
