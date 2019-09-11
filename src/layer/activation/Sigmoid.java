package layer.activation;

/**
 * Sigmoid activation function applies the function: 1 / (1 + exp(-x)), where
 * x is the input. This activation function has an S shaped form over the range
 * (0,1). Using an inner product layer + this function in a feed forward neural
 * net is equivalent to running a logistic regression (for the case of one
 * output).
 * 
 * The sigmoid activation function is not recommended for deep neural nets. The
 * trouble with sigmoid activation function is that the gradient has a maximum
 * of 0.25 (at 0). This means that when there are several layers the gradient
 * vanishes for the layers closest to the input. In such cases ReLU is better
 * since gradient = 1.0 for any value bigger than 0.0.
 * 
 * @author diego_paez
 */
public class Sigmoid implements ActivationFunction{

    /**
     * Apply sigmoid function: 1 / (1 + exp(-x)).
     * @param input Input to the sigmoid function.
     * @param isTest Whether this is a test or training set (not used).
     * @return Sigmoid.
     */
    @Override
    public double forward(double input, boolean isTest) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    /**
     * Calculate derivative of sigmoid = s(x) * (1 - s(x).
     * @param input Input to the sigmoid (not used).
     * @param output Output of the sigmoid.
     * @return Derivative of input with respect to output.
     */
    @Override
    public double derivative(double input, double output) {
        return (1 - output) * output;
    }

}
