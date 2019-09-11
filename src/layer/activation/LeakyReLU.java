package layer.activation;

/**
 * This is a type of activation function based on the ReLU, to tackle the
 * dying ReLU problem (ReLUs with input smaller than zero deactivate). To deal
 * with this problem negative inputs have a small output = input * alpha
 * where alpha is smaller than 1.0, but not zero as in ReLU). See articles:
 * - "Empirical Evaluation of Rectified Activations in Convolutional Network"
 * - "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
 * Classification"
 * by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
 * @author diego_paez
 */
public class LeakyReLU implements ActivationFunction {

    /**
     * Constant by which input will be multiplied in case it is smaller than
     * zero. The default value is 1.0 / 5.5 = 0.1818...
     */
    private double alpha = 1.0 / 5.5;

    /**
     * Create LeakyReLU activation function with default value of 1.0 / 5.5.
     */
    public LeakyReLU() {
    }

    /**
     * Create LeakyReLU activation function with value of alpha as given.
     * @param alpha Weight to give to negative inputs.
     */
    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }

    /**
     * Apply LeakyReLU function which = input if positive and input * alpha
     * if negative.
     * @param input Value to apply LeakyReLU function.
     * @param isTest Whether it is test data or not, this parameter is not used
     * but inherited from interface (not used).
     * @return LeakyReLU output = input or input * alpha. 
     */
    @Override
    public double forward(double input, boolean isTest) {
        if(input > 0)
            return input;
        else
            return input * alpha;
    }

    /**
     * Derivative of LeakyReLU = 1.0 if positive, and = alpha if negative.
     * @param input Input of this activation function.
     * @param output Ouptut of LeakyReLU.
     * @return Derivative of output with respect to input.
     */
    @Override
    public double derivative(double input, double output) {
        if(output > 0)
            return 1.0;
        else
            return alpha;
    }

}
