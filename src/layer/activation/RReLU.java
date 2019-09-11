package layer.activation;

import java.util.Random;

/**
 * RReLU is short for Randomized Rectified Linear Unit and works similar
 * to Leaky ReLU in the sense that tries to deal with dying ReLU problem.
 * The activation function returns the input if positive and the input times
 * a small value if negative. This small value is random if the data is test data
 * else it is a constant. Both values, test and training depend on a lower limit
 * value and a upper limit value.
 * 
 * "Empirical Evaluation of Rectified Activations in Convolutional Network"
 * @author diego_paez
 */
public class RReLU implements ActivationFunction{

    /**
     * Low value, default = 1.0 / 8.0 = 0.125.
     */
    private double l = 1.0 / 8.0;
    
    /**
     * Upper value, default = 1.0 / 3.0 = 0.333.
     */
    private double u = 1.0 / 3.0;
    
    /**
     * Value by which negative values are multiplied.
     */
    private double a;
    
    /**
     * Random values generator.
     */
    private final Random rand;

    
    /**
     * Initialize RReLU, use default values for u and l.
     */
    public RReLU() {
        rand = new Random();
    }

    /**
     * Initialize RReLU, use specified seed for random generator.
     * @param seed Seed for random generator.
     */
    public RReLU(int seed) {
        rand = new Random(seed);
    }

    /**
     * Initialize RReLU, use specified seed, u and l values.
     * @param l Lower value.
     * @param u Upper value.
     * @param seed Seed for random generator.
     */
    public RReLU(double l, double u, int seed) {
        this.l = l;
        this.u = u;
        rand = new Random(seed);
    }

    /**
     * Forward propagate input given to this function. If positive return input,
     * if negative return depending of whether if training or test:
     * If training data then return input * middle value between l and u.
     * If test data then return input * a random value between l and u.
     * @param input
     * @param isTest
     * @return 
     */
    @Override
    public double forward(double input, boolean isTest) {
        if(!isTest)
            a = rand.nextDouble() * (u - l) + l;
        else
            a = (l + u) / 2.0;

        if(input > 0)
            return input;
        else
            return input * a;
    }

    /**
     * Derivative of RReLU, 1.0 if positive input, a = value that input is multiplied
     * by if input is negative.
     * @param input Input value of this layer.
     * @param output Output value of this layer.
     * @return Derivative of output w.r. to input.
     */
    @Override
    public double derivative(double input, double output) {
        if(output > 0)
            return 1.0;
        else
            return a;
    }

}
