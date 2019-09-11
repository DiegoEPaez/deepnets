package core;

import optim.BatchFunction;
import tensor.DoubleTensor;

/**
 * Function of a feed forward neural net. This function is optimized to find
 * the best possible weights of a feed forward net.
 * @author diego_paez
 */
public class NeuralNetFunction implements BatchFunction{

    
    /**
     * Contains all info related to the neural net.
     */
    public NeuralNetModel model;

    /**
     * Create a neural net function with the specified neural net model.
     * @param model 
     */
    public NeuralNetFunction(NeuralNetModel model) {
        this.model = model;
    }
    
    /**
     * For a given point (value), evaluate using the data suplied. The use of
     * passing the data XBatch, yBatch and yWeightsBatch is to be able to submit
     * the data in batches to allow algorithms such as stochastic gradient descent
     * to evaluate the function in parts.
     * @param value Point which will be valued with current neural net (model).
     * @param XBatch Current batch of training data
     * @param yBatch Current batch of target data
     * @param yWeightsBatch Current batch of weights to be assigned to each
     * example
     * @param lambdaAdjus Lambda adjustment due to the use of regularization.
     * When using batches of smaller size than the whole training set the
     * regularization must be adjusted since the cost function (without reg.9)
     * will be smaller for smaller batches.
     * @return Total cost for the neural net function.
     */
    @Override
    public double value(double[] value, DoubleTensor XBatch, DoubleTensor yBatch,
            DoubleTensor yWeightsBatch, double lambdaAdjus) {
        
        // Set the weights of the neural net in the model (the values are copied
        // into the data structures (DoubleTensor) of the neural net.
        model.setWeights(value);
        
        // calculate cost & return it
        double t = model.cost(XBatch, yBatch, yWeightsBatch,lambdaAdjus);
        return t;
    }

}
