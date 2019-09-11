package layer.activation;

import layer.Layer;
import tensor.DoubleTensor;

/**
 * This layer applies a real to real function (activation function) to each input.
 * Some examples of activation functions are exponential, hyperboic tangent, and
 * sigmoid.
 * @author diego_paez
 */
public class ActivationLayer extends Layer{

    /**
     * The activation function that will be applied in this layer.
     */
    public ActivationFunction activationFun;

    /**
     * Requires the activation function to be able to apply it to all inputs.
     * @param activationFun The activation function.
     */
    public ActivationLayer(ActivationFunction activationFun){
        super();
        this.activationFun = activationFun;
    }

    /**
     * Forward propagates inputs by applying the activation function.
     * @param input The inputs to this layer.
     * @param isTest Whether the data given is test data (true) or training data
     * (false)
     * @return Output tensor (result of applying activation function to inputs). 
     */
    @Override
    public DoubleTensor fProp(DoubleTensor input, boolean isTest) {
        final int sz = input.size();
        this.input = input;
        double ii;

        for(int i = 0; i < sz;i++){
            ii = input.data.get(i);
            output.data.set(i, activationFun.forward(ii, isTest));
        }

        return output;
    }

    /**
     * Calculates the chainGrad (derivative of loss function with respect to
     * inputs of this layer). The calculation as with most layers is the product
     * of the incoming chaingrad (derivative of loss function with respect to
     * outputs), times the derivative of the activation function (derivative of
     * outputs with respect to inputs). The theorem applied is the chain rule
     * theorem of calculus.
     *
     * @param chainGrad Derivative of loss function with respect to outputs.
     * @return Derivative of loss function with respect to inputs.
     */
    @Override
    public DoubleTensor bProp(DoubleTensor chainGrad) {
        double oi, ii;
        for (int i = 0; i < chainGrad.size(); i++){
            oi = output.data.get(i);
            ii = input.data.get(i);
            grad.data.set(i, activationFun.derivative(ii, oi) * chainGrad.getQuick(i));
        }
        return grad;
    }
}
