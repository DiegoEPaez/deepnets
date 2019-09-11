package layer.activation;

/**
 * Specifies the basic behavior an activation function of an activation layer
 * must have. Particulary it must be able to transform a given real number into
 * another real number (by applying a certain function). Secondly be able to
 * calculate the derivative of the function.
 * @author diego_paez
 */
public interface ActivationFunction {

    /**
     * This function is the activation function that is intended to be applied
     * to a activation layer.
     * @param input A real number (or double precision floating point number to
     * be more precise) that the activation function will be applied to.
     * @param isTest Whether the input data is test data (true) or training data
     * (fals)
     * @return The result of applying the activation function to the given input. 
     */
    public double forward(double input, boolean isTest);
    
    /**
     * Calculates the derivative of the output with respect to the input. The
     * derivative could in principle be calculated just with the input, but to
     * make calculations faster, sometimes it is quicker to use the output of
     * the function, such is the case of the sigmoid activation function but not
     * the case of SoftSign (or at least not as of the implementation of this
     * writing).
     * @param input The given input to this layer.
     * @param output The output of this layer.
     * @return Derivative of output with respect to input. 
     */
    public double derivative(double input, double output);
}
