package loss;

import tensor.DoubleTensor;
import tensor.TensorFunctions;

/**
 *
 * @author diego_paez
 */
public class WeightedCrossEntropy implements LossFunction{

    /**
     * WCE = -sum(I(y) .* log(nnOut) .* sW) = -sum(vectorByIndex(log(nnOut + eps)) .* sW)
     *
     * where
     * I(y) = if y is a row vector, I(y) is a matrix with a 1 in
     *  position a_y(i)_i = MatrixUtil.OneIfEqualsMat(y,max_label)
     * sW = sample weights
     * 
     * @param output
     * @param y
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

        // !!multiply entries of temp matrix by weight vector beta
        temp.muli(yWeights);

        double fun = -1.0 * temp.sum();
        return fun;
    }

    /**
     * grad WCE = -I(labels).* 1./nnOut .* sW (returns a matrix,though each column only
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
        DoubleTensor temp = output.add(TensorFunctions.smallConst).rdivi(1.0);
        DoubleTensor temp2 = y.index(output.dims[0]);
        temp.muli(temp2);
        temp.muliLowerDimTensor(0, yWeights);

        return temp.muli(-1.0);
    }

}
