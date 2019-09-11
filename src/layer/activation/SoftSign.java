package layer.activation;

/**
 * Activation function with range over (-1,1) which can be considered an alternative
 * to tanh. (see http://cs224d.stanford.edu/lecture_notes/LectureNotes3.pdf).
 * @author diego_paez
 */
public class SoftSign implements ActivationFunction{

    /**
     * Apply SoftSign activation function = x / ( 1 + |x|).
     * @param input Input to SoftSign.
     * @param isTest Whether data is test data or training data.
     * @return Result of applying softsign.
     */
    @Override
    public double forward(double input, boolean isTest) {
        return input / (1.0 + Math.abs(input));
    }

    /**
     * Calculate derivative of SoftSign = 1 / (1 + |x|)^2.
     * @param input Input to SoftSign activation function.
     * @param output Output of SoftSign.
     * @return Derivative of SoftSign.
     */
    @Override
    public double derivative(double input, double output) {
        double num = (1.0 + Math.abs(input));
        num *= num;
        return 1.0 / num;
    }

}
