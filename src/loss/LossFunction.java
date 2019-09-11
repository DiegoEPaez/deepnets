package loss;

import tensor.DoubleTensor;

/**
 *
 * @author diego_paez
 */
public interface LossFunction {

    public double eval(DoubleTensor output, DoubleTensor y, DoubleTensor yWeights);
    public DoubleTensor bProp(DoubleTensor output, DoubleTensor y, DoubleTensor yWeights);
}
