package init;

/**
 * Interface for weight initialization methods.
 * @author diego_paez
 */
public interface Initialization {
    
    /**
     * Init biases.
     * @param numBias Number of biases to init.
     * @return Array of size[numBias]
     */
    public abstract double[] initBias(int numBias);
    
    /**
     * Init weights.
     * @param numInputs Number of inputs to the layer.
     * @param numOutputs Number of outputs to the layer
     * @param dimWeights Dimensions of the weight tensor.
     * @return Array of size[Mult(dims)]
     */
    public abstract double[] initWeights(int numInputs, int numOutputs, int... dimWeights);
}
