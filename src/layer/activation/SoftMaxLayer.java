package layer.activation;

import layer.Layer;
import org.jblas.NativeBlas;
import tensor.DoubleTensor;
import tensor.TensorFunctions;

/**
 * Apply softmax function. The softmax function creates a probability distribution
 * over all outputs by normalizing them.
 * @author diego_paez
 */
public class SoftMaxLayer extends Layer{
    
    /**
     * Number of inputs in one dimension, without examples. That is how many
     * inputs there are not taking into account examples. Note that there may
     * be several input dimensions in that case, all dimensions (except examples)
     * are multiplied to arrive to the nI1D value.
     */
    protected int nI1D;
    
    /**
     * Set the dimension of inputs, also calculate nI1D.
     * @param inputs 
     */
    @Override
    public void setDimsOfInputsWOE(int... inputs) {
        nI = inputs;
        nI1D = 1;

        for(int i = 0; i < nI.length; i++){
            nI1D *= inputs[i];
        }
    }
   
    /**
     * Given a matrix v = [v1;v2;...;vn], where vi is the ith row, and represents
     * a different input (examples are by columns).
     * Softmax function evaluates in the following way:
     *
     * softmax(v) = [exp(v1) / sum(exp(vi),i=1..n); exp(v2) / sum(exp(vi),i=1..n);...]
     *
     * Contrary to other functions softmax is not a function R->R but rather can be
     * seen as a function R^n->R^n (n is the number outputs).
     * Though it is only used once per example, contrary to sigmoid for example
     * that would be used n times (assuming n outputs) per column(example).
     *
     * Big numbers within the input matrix can cause problems because:
     * exp(big_number) = infinity and Infinity / Infinity = NaN
     * And thus when normalizing the denominator = infinity the result will
     * be NaN. This is prone to happen when one of the variables
     * has very high probability of ocurring.
     *
     * Small numbers are not a problem because:
     * exp(small_number) = 0.0
     *
     * To avoid this problem the max per column is subtracted to each column,
     * because for a given column with values [a1;a2;...;an]
     *
     * exp(ai) / sum(exp(ai),i = 1...n) = exp(ai - aM) / sum(exp(ai - aM), i = 1...n)
     *
     * where aM = max(ai, i = 1...n)
     *
     * The problem with this approach is that some precision might be lost due to
     * having smaller numbers (but this is the approach used).
     *
     * Thus result is a matrix where softmax:R^n->R^n is applied once per example.
     * @param input Tensor of any number of dimensions, when calculating the
     * softmax function every dimension except last one is compressed into a single
     * dimension. Generally the input shoud be a matrix: numOfVars X examples
     * @param isTest Whether it is test or training data (not used).
     * @return Softmax activation.
     */
    @Override
    public DoubleTensor fProp(DoubleTensor input, boolean isTest) {
        this.input = input;
        
        // compress all dimensions except last into a single one.
        int[] dims = input.dims;
        int[] dims2d = DoubleTensor.shapeDims2d(input.dims);
        input.reshape(dims2d);
        
        DoubleTensor rowNorm, maxCol;

        // subtract max per column to avoid problem with Infinity.
        maxCol = input.byDimMax(0);
        input.subiLowerDimTensor(output, 0, maxCol); // subi & place on output

        // Now calculate softmax function
        TensorFunctions.expi(output);

        // create vector with the sum by columns
        rowNorm = output.byDimSum(0);

        // normalize inputs
        output.diviLowerDimTensor(0, rowNorm);

        // reshape input to original dims
        input.reshape(dims);
        
        return output;
    }

    /**
     * Calculates the chaingrad of the Softmax function. That is the derivative
     * of the loss function with respect to the inputs of the Softmax function.
     * Although both Softmax and InnerProduct layers are functions of R^n to R^n,
     * the chaingrad of the latter is easier to calculate because the derivative
     * of the InnerProduct w.r. to its inputs is always a constant = theta, thus
     * theta can be matrix multiplied by the incoming chaingrad (of the next layer
     * or loss function). For the case of the Softmax, the Softmax w.r. to
     * its inputs differs per example and calculating the chaingrad of the Softmax
     * entails multiplying each matrix (obtained by derivating the Softmax w.r. 
     * to its inputs, for each example) times the corresponding vector of the
     * incoming chaingrad (i.e., each subtensor of the last dimension - each
     * column of a 2D tensor /matrix).
     *
     * A shortcut can be taken when using crossentropy with softmax
     * (See gradCE_Softmax method of NeuralNetModel)
     *
     * For the jth example the softmax derivative w.r. to its inputs is (s = softmax func.):
     * 
     * s(a0j) * ( 1 - s(a0j))| -s(a0j) * s(a1j)     | -s(a0j) * s(a2j) ...
     * -s(a1j) * s(a0j)      | s(a1j) * (1 - s(a1j) | -s(a1j) * s(a2j) ...
     * -s(a2j) * s(a0j)      | -s(a2j) * s(a1j)     | s(a2j) * (1 - s(a2j)
     *
     * @param chainGrad Tensor of any number of dimensions, when calculating
     * the gradient it is reshaped into 2 dimensions. Generally the input is
     * numOfVars X Examples.
     * @return numOfVars X Examples
     */
    @Override
    public DoubleTensor bProp(DoubleTensor chainGrad) {
        // Used to store the derivative of the softmax w.r. to its inputs        
        int[] dimS = new int[]{nI1D,nI1D};
        DoubleTensor s = new DoubleTensor(dimS);
        
        // used to calculate the offset of the chainGrad tensor when moving through
        // each example (column)
        int offset;
        
        // Iterate over each example
        for(int i = 0; i < chainGrad.lastDim(); i++){
            // Get the derivative of the softmax w.r. to its inputs
            offset = i * nI1D;
            softWRInp(s,offset);
            
            // s * ith col of chainGrad, store in ith col of grad
            NativeBlas.dgemm('N', 'N', nI1D, 1, nI1D, 1.0, s.data.getData(),
                    0, nI1D, chainGrad.data.getData(), offset,
                    nI1D, 0.0, grad.data.getData(), offset, nI1D);
        }
        return grad;
    }
    
    /**
     * Calculate softmax w.r. to inputs.
     * @param s Where to store the derivative.
     * @param offset Offset for the output tensor.
     */
    private void softWRInp(DoubleTensor s,int offset){
        double aux;
        int ind = 0;
        for(int j = 0; j < nI[0]; j++){
            for(int k = 0; k < nI[0]; k++){
                if(j == k){
                    aux = output.getQuick(j + offset) * ( 1 - output.getQuick(j + offset));
                } else{
                    aux = -output.getQuick(j + offset) * output.getQuick(k + offset);
                }
                s.setQuick(ind, aux);
                ind++;
            }
        }
    }
}