package layer;

import tensor.DoubleTensor;

/**
 * This class defines what a feed forward neural net should have at a minimum.
 * Since this class is abstract it must be inherited and abstract methods
 * implemented (forward propagation and backward propagation) to be used.
 * @author diego_paez
 */
public abstract class Layer {

    /**
     * Number of inputs (dimensions), not including examples.
     */
    public int[] nI;
    
    /**
     * Input tensor for last forward propagation.
     */
    protected DoubleTensor input;
    
    /**
     * Output tensor for last forward propagation.
     */
    protected DoubleTensor output;
    
    /**
     * Gradient calculated during last backward propagation.
     */
    protected DoubleTensor grad;

    /**
     * Set dims of inputs not counting examples.
     * @param inputs Dims of inputs.
     */
    public void setDimsOfInputsWOE(int... inputs) {
        nI = inputs;
    }

    /**
     * Get dims of outputs not counting examples. In some cases outputs = inputs.
     * @return Number of outputs without examples.
     */
    public int[] getDimsOfOutputsWOE() {
        return nI;
    }


    /**
     * During each pass of the feed forward neural net, the number of examples
     * may differ, thus each time the feed forward neural net is run this
     * method should be called. Generally the output and grad vectors will
     * remain as they are (call to method resize).
     * @param numExamples Number of examples that will pass through the neural
     * net.
     */
    public void initSpaceInMemory(int numExamples) {
        int[] dimsOut = new int[nI.length + 1];
        int len = 1;
        for(int i = 0; i < nI.length;i++){
            dimsOut[i] = nI[i];
            len *= nI[i];
        }
        dimsOut[nI.length] = numExamples;
        len *= numExamples;

        if(output == null){
            output = new DoubleTensor(dimsOut);
            output.setPosToLast();
            grad = new DoubleTensor(dimsOut);
            grad.setPosToLast();
        } else{
            output.resize(dimsOut);
            output.setPos(len);
            grad.resize(dimsOut);
            grad.setPos(len);
        }
    }
    
    /**
     * Return the output tensor of this layer.
     * @return Output Tensor.
     */
    public DoubleTensor getOutput(){
        return output;
    }

    /**
     * Abstract method that describes the forward propagation method of a neural
     * net to be implemented by any layer. The layer should take the input
     * tensor and transform it in some way to generate the output tensor.
     * @param input The input tensor incoming from a previous layer or input data.
     * @param isTest Whether the test set is running or not, some layers behave
     * different depending on whether the data is training or test data.
     * @return Output tensor.
     */
    public abstract DoubleTensor fProp(DoubleTensor input, boolean isTest);
    
    /**
     * Abstract method that describes the back propagation method of a neural
     * net to be implemented by any layer. The layer should take the incoming
     * chaingrad (derivative of the loss function with respect to the inputs of
     * next layer = outputs of this layer) and produce the chaingrad with respect
     * to the inputs of this layer.
     * 
     * NOTE:
     * dJ(.) / dzj_l = delta in Andrew Ng's notes = chaingrad
     * daj_l / dzj_l = f'(z) = derivative of this layer (derivative of outputs
     * w.r. to inputs).
     * dJ(.) / dzj_l = (daj_l / dzj_l) * (dJ(.) / daj_l) ...(1) = equation
     * that links chaingrads with derivative of this layer.
     *
     * Where
     * J is the loss function.
     * aj_l is the jth output of the lth layer.
     * zj_l is the jth input of the lth layer.
     *
     * @param chainGrad Derivative of the loss function with respect to the outputs
     * of this layer.
     * @return Derivative of the loss function with respect to the inputs of
     * this layer.
     */
    public abstract DoubleTensor bProp(DoubleTensor chainGrad);
}
