package layer.activation;

/**
 * This is an identity activation function meaning that inputs are forward propagated
 * in the same way they enter the neural net,i.e., f(x) = x. This function is completely
 * useless in a neural net, but I am going to leave it as reference. It is
 * completeley useless, since it has no impact whatsoever in the net inputs pass
 * as they are, chaingrad is backpropagated as is (mult. by one leaves chaingrad
 * the same).
 * @author diego_paez
 */
public class Identity implements ActivationFunction{

    /**
     * Pass input as is.
     * @param input Input to function.
     * @param isTest Whether test o training set (not used).
     * @return input
     */
    @Override
    public double forward(double input, boolean isTest) {
        return input;
    }

    /**
     * Derivative of identity function = 1.0.
     * @param input Input of this layer.
     * @param output Output of this layer (= input).
     * @return Derivative of output w.r. to input.
     */
    @Override
    public double derivative(double input, double output) {
        return 1.0;
    }

}
