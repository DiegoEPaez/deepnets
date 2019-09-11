package init;

import java.util.Arrays;
import java.util.Random;

/**
 * Initialization of weights of a layer of a neural net using the method described
 * in the article:
 * "Understanding the difficulty training deep feedforward nn:"
 * by Xavier Glorot and Yoshua Bengio.
 * 
 * @author diego_paez
 */
public class Xavier implements Initialization{

    /**
     * Random numbers generator.
     */
    public Random rand = new Random();

    public Xavier() {
    }

    /**
     * Init. the random generator with given seed
     * @param seed 
     */
    public Xavier(int seed) {
        rand = new Random(seed);
    }


    /**
     * Initialize all biases to zero (following the article).
     * @param numBias number of biases
     * @return array with zeros.
     */
    @Override
    public double[] initBias(int numBias) {
        double[] bias = new double[numBias];
        Arrays.fill(bias, 0.0);

        return bias;
    }

    /**
     * Initialize weights by using a uniform standard distribution around zero
     * with specified width:
     * 
     * Unif[-sqrt(6) / sqrt(size[0] + size[1]),sqrt(6) / sqrt(size[0] + size[1])]
     * 
     * @param numInputs Number of inputs of this layer.
     * @param numOutputs Number of outputs of this layer.
     * @param dims Dimensions for the weight Tensor.
     * @return array with weights.
     */
    @Override
    public double[] initWeights(int numInputs, int numOutputs, int... dims) {
        // "xavier" initialization in DeepLearning4j
        // Use "Understanding the difficulty training deep feedforward nn:"
        // Unif[-sqrt(6) / sqrt(size[0] + size[1]),sqrt(6) / sqrt(size[0] + size[1])]
        // Get uniform values between -1 and +1
        int size = 1;
        for(int i = 0; i < dims.length; i++){
            size *= dims[i];
        }
        
        double factor = Math.sqrt(6.0 / (numInputs + numOutputs));
        double[] result = new double[size];

        for(int i = 0; i < size; i++){
            result[i] = 2.0 * rand.nextDouble() - 1.0;
            result[i] *= factor;
        }

        return result;
    }

}
