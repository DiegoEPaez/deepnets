package init;

import java.util.Arrays;
import java.util.Random;
import org.apache.commons.math3.distribution.NormalDistribution;

/**
 * Initialization of weights of a layer of a neural net using the method described
 * in the article:
 * "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
 * by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
 * 
 * @author diego_paez
 */
public class He implements Initialization{

    /**
     * Normal distribution object.
     */
    public NormalDistribution norm = new NormalDistribution();
    
    /**
     * Random numbers generator.
     */
    public Random rand = new Random();

    public He() {
    }

    /**
     * Init. the random generator with given seed
     * @param seed 
     */
    public He(int seed) {
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
     * Initialize weights by using a normal standard distribution and mutltiplying
     * by the factor in He's paper (after eq 10), so that the normal distribution
     * has desired variance.
     * @param numInputs Number of inputs of this layer.
     * @param numOutputs Number of outputs of this layer.
     * @param dims Dimensions for the weight Tensor.
     * @return array with weights.
     */
    @Override
    public double[] initWeights(int numInputs, int numOutputs, int... dims) {
        // "ReLU" initialization in DeepLearning4j
        // Use "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        int size = 1;
        for(int i = 0; i < dims.length; i++){
            size *= dims[i];
        }

        double factor = Math.sqrt(2.0 / numInputs);
        double[] result = new double[size];

        for(int i = 0; i < size; i++){
            result[i] = norm.inverseCumulativeProbability(rand.nextDouble());
            result[i] *= factor;
        }

        return result;
    }

}
