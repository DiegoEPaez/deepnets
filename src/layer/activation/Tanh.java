package layer.activation;

/**
 * Activation function with range over (-1,1). Tanh has be shown to perform
 * converge faster in practice than sigmoid function due to the fact that
 * its output is symmetric (from -1 to 1), contrary to sigmoid's output of (0,1).
 * ReLU has been shown to converge even faster. 
 * @author diego_paez
 */
public class Tanh implements ActivationFunction{

    /**
     * Applies tanh to input.
     * @param input Input to which apply tanh.
     * @param isTest Whether test or training data (not used).
     * @return Tanh of input.
     */
    @Override
    public double forward(double input, boolean isTest) {
        return Math.tanh(input);
    }

    /**
     * Gradient of tanh(x) = 1 - tanh^2(x)
     * @param input Input to this activation function (not used).
     * @param output Output of activation function = tanh.
     * @return Derivative of tanh.
     */
    @Override
    public double derivative(double input, double output) {
        return 1 - output * output;
    }

}
