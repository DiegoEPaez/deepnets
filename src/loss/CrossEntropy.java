package loss;

import tensor.DoubleTensor;
import tensor.TensorFunctions;

/**
 * Note:
 * Using something different than softmax, like sigmoid, with
 * cross entropy does not make much sense since the value of the other classes for the
 * given example can take any value and are not penalized.
 * @author diego_paez
 */
public class CrossEntropy implements LossFunction{

    /**
     * CE = -sum(I(y) .* log(nnOut + eps)) = -sum(vectorByIndex(log(nnOut + eps)))
     *
     * where
     * I(y) = if y is a row vector, I(y) is a matrix with a 1 in
     *  position a_y(i)_i = MatrixUtil.OneIfEqualsMat(y,max_label)
     * eps = small constant to avoid zeros
     * @param output matrix with predictions in each column for each example. The
     * matrix has different rows for every class.
     * @param y row vector with correct values for each example
     * @param yWeights
     * @return
     */
    @Override
    public double eval(DoubleTensor output, DoubleTensor y, DoubleTensor yWeights) {

        // Obtain values as indicated by y vector (select by rows = dim 0)
        DoubleTensor temp = output.getLowerDim(0, y);

        // Obtain the logarithm of the matrix plus smallConst
        // special care must be taken with derivative!!, smallConst should not be forgotten
        temp.addi(TensorFunctions.smallConst);
        TensorFunctions.logi(temp);

        double fun = -1.0 * temp.sum();
        return fun;
    }

    /**
     * grad CE = -I(y).* 1./ (nnOut + eps) (returns a matrix,though each column only
     * has a positive value and the rest = 0). It returns a matrix because
     * crossEntropy: R^nxm -> R. Thus gradient is a matrix of size n x m
     *
     * @param output
     * @param y
     * @param yWeights
     * @return
     */
    @Override
    public DoubleTensor bProp(DoubleTensor output, DoubleTensor y, DoubleTensor yWeights) {
        DoubleTensor temp = output.add(TensorFunctions.smallConst);
        temp.rdivi(1.0);
        DoubleTensor temp2 = y.index(output.dims[0]);
        temp.muli(temp2);

        return temp.muli(-1.0);
    }

}
