package core;

import optim.BatchGradient;
import tensor.DoubleTensor;

/**
 * Calculates gradient of a feed forward neural net. Required to optimize the
 * weights of a feed forward neural net.
 * @author diego_paez
 */
public class NeuralNetGradient implements BatchGradient{
    
    /**
     * Contains all info related to the neural net.
     */
    public NeuralNetModel model;

    /**
     * Create a neural net function with the specified neural net model.
     * @param model 
     */
    public NeuralNetGradient(NeuralNetModel model) {
        this.model = model;
    }

    /**
     * For a given point (value), obtain the gradient with data suplied. The use
     * of passing the data XBatch, yBatch and yWeightsBatch is to be able to submit
     * the data in batches to allow algorithms such as stochastic gradient descent
     * to evaluate the function in parts.
     * @param value Point which will be valued with current neural net (model).
     * This parameter is not really used, rather it is set in the model when
     * forward propagation (in NeuralNetFunction). Since this class implements
     * Batch Gradient, this parameter must be included. Copying the value parameter
     * again woud slow the performance of the neural net.
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
    public double[] value(double[] value, DoubleTensor XBatch, DoubleTensor yBatch,
            DoubleTensor yWeightsBatch, double lambdaAdjus) {
        
        // Use model's back propagation algorithm
        model.bProp(yBatch, yWeightsBatch,lambdaAdjus);
        
        // Return the gradient of the weights
        return model.thetaGrad;
    }

}
