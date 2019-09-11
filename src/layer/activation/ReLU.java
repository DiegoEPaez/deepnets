package layer.activation;

/**
 * ReLU or Rectified Linear Unit, returns the maximum between zero and the input.
 * @author diego_paez
 */
public class ReLU implements ActivationFunction{

    /**
     * Return maximum between zero and input.
     * @param input Input to ReLU.
     * @param isTest Whether test set or train set (not used).
     * @return Maximum between zero and input.
     */
    @Override
    public double forward(double input, boolean isTest) {
        return Math.max(0, input);
    }

    /**
     * Derivative of ReLU 0.0 if negative input, 1.0 if positive.
     * @param input Input to this unit.
     * @param output Output of this unit.
     * @return Return 1.0 if negative input 0.0 if positive input.
     */
    @Override
    public double derivative(double input, double output) {
        return output > 0.0 ? 1.0: 0.0;
    }

}
