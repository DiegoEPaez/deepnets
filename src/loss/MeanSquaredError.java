package loss;

import tensor.DoubleTensor;
import tensor.TensorFunctions;

/**
 *
 * @author diego_paez
 */
public class MeanSquaredError implements LossFunction{

    /**
     * MSE = 1/ m * sum((h(X)(i)- y(i)).^2,i=1...m)
     * @param nnOut matrix with predicitons in each column for each example. The
     * matrix has different rows for every class.
     * @param output
     * @param y
     * @param yWeights
     * @return
     */
    @Override
    public double eval(DoubleTensor output, DoubleTensor y, DoubleTensor yWeights) {
        DoubleTensor temp;

        temp = output.sub(y);
        TensorFunctions.powi(temp, 2.0);

        return temp.sum() / temp.size();
    }

    /**
     * grad MSE = 2.0 / m * (h(X) - y(i))
     *
     * @param output
     * @param y
     * @param yWeights
     * @return
     */
    @Override
    public DoubleTensor bProp(DoubleTensor output, DoubleTensor y, DoubleTensor yWeights) {
        DoubleTensor temp;

        temp = output.sub(y);

        return temp.muli(2.0 / temp.size());
    }

}
