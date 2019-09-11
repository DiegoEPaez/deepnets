package layer.activation;

/**
 * Activation function which mimics the ReLU activation function except that
 * the function is thought out so it is derivable at every point (contrary
 * to ReLU which is non derivable at 0). The softplus function is:
 * 
 * ln(1 + exp(x))
 * @author diego_paez
 */
public class SoftPlus implements ActivationFunction{

    /**
     * Constant such that 1 + exp(-limExp) = 1.0 in double arithmetic.
     */
    public static final double LIMEXP = 36.0436533891172;

    /**
     * Calculate softplus function: ln(1 + exp(x)).
     * @param input Input to softplus function.
     * @param isTest Whether test or traing data is given (not used).
     * @return Softplus function of input.
     */
    @Override
    public double forward(double input, boolean isTest) {
        // if input is bigger than LIMEXP just return input (else will return
        // infinity which will derive in the whole calculation being inf. which is wrong)
        // something similar applies to -LIMEXP
        if(input > LIMEXP){
            return input;
        } else if(input < -LIMEXP){
            return 0.0;
        } else {
            return Math.log(1.0 + Math.exp(input));
        }
    }

    /**
     * Calculate derivative of softplus = 1 / (1 + exp(-x)).
     * @param input Input to this layer.
     * @param output Output of softplus function.
     * @return Derivative of output w.r. to input.
     */
    @Override
    public double derivative(double input, double output) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

}
