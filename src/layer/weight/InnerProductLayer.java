package layer.weight;

import init.Initialization;
import tensor.DoubleTensor;

/**
 *
 * @author diego_paez
 */
public class InnerProductLayer extends WeightLayer {

    // number of neurons
    protected int nN;
    
    /**
     * Number of inputs in one dimension, without examples. That is how many
     * inputs there are not taking into account examples. Note that there may
     * be several input dimensions in that case, all dimensions (except examples)
     * are multiplied to arrive to the nI1D value.
     */
    protected int nI1D;

    public InnerProductLayer(Initialization init, int numNeurons){
        super();
        this.init = init;
        this.nN = numNeurons;
    }

    @Override
    public void setDimsOfInputsWOE(int... inputs) {
        nI = inputs;
        nI1D = 1;

        for(int i = 0; i < nI.length; i++){
            nI1D *= inputs[i];
        }
    }

    @Override
    public int[] getDimsOfOutputsWOE() {
        return new int[]{nN};
    }

    @Override
    public final void initParams() {

        bias = new DoubleTensor(init.initBias(nN), nN);
        biasGrad = new DoubleTensor(bias.dims);
        biasGrad.data.setPos(biasGrad.data.getData().length);

        weights = new DoubleTensor(init.initWeights(nI1D,nN, new int[]{nI1D,nN}),
                new int[]{nI1D,nN});
        weightsGrad = new DoubleTensor(weights.dims);
        weightsGrad.data.setPos(weightsGrad.data.getData().length);
    }

    @Override
    public void initSpaceInMemory(int numExamples){
        int[] ins = new int[nI.length + 1];
        System.arraycopy(nI, 0, ins, 0, nI.length);
        ins[nI.length] = numExamples;

        if(output == null){
            output = new DoubleTensor(nN,numExamples);
            output.setPosToLast();

            grad = new DoubleTensor(ins);
            grad.setPosToLast();
        } else{
            output.resize(nN,numExamples);
            output.setPos(nN * numExamples);
            grad.resize(ins);
            grad.setPos(nI1D * numExamples);
        }
    }

    @Override
    public int getNumberOfParams() {
        return (nI1D + 1) * nN;
    }

    /**
     * Receives a DoubleTensor and reshapes it to be in 2D assuming first dim
     * are examples, then matrix multiplies by weights.
     * @param input
     * @param isTest
     * @return
     */
    @Override
    public DoubleTensor fProp(DoubleTensor input, boolean isTest) {
        this.input = input;
        int[] dims = input.dims;
        int[] dims2d = DoubleTensor.shapeDims2d(input.dims);
        input.reshape(dims2d);
        
        output = weights.mmuli(true, false, input,output);

        // add bias
        output.addiLowerDimTensor(1, bias);
        
        // reshape input to original dims
        input.reshape(dims);
        
        return output;
    }

    @Override
    public DoubleTensor bProp(DoubleTensor chainGrad) {
        // middle dim = number of outputs
        return weights.mmuli(false,false,chainGrad,grad);
    }

    @Override
    public void updateLayerWGrad(DoubleTensor chainGrad) {
        int[] dims = input.dims;
        int[] dims2d = DoubleTensor.shapeDims2d(input.dims);
        input.reshape(dims2d);
        
        // Matrix multiply to apply chain rule (middle dim. = examples)
        input.mmuli(false, true, chainGrad, weightsGrad);
        

        chainGrad.byDimSumi(chainGrad.dims.length - 1, biasGrad);
        
        input.reshape(dims);
    }

    
}
