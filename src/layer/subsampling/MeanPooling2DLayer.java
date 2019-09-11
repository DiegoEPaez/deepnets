package layer.subsampling;

import layer.Layer;
import org.apache.log4j.Logger;
import tensor.DoubleTensor;

/**
 * Applies mean pooling to a given tensor. The mean pooling is done by getting
 * the mean of every pW x pH rectangle and moving sW or sH units to next rectangle.
 * @author diego_paez
 */
public class MeanPooling2DLayer extends Layer{

    /**
     * Logger of log4j.
     */
    private static final Logger LOG = Logger.getLogger(MeanPooling2DLayer.class);

    /**
     * Number of channels every channel contains a set of inputs of 2 dimensions.
     * For instance there could be 3 channels corresponding to red, green and 
     * blue colors.
     */
    protected int nC;

    /**
     * Width of given input. For example could be number of pixels in an image.
     */
    protected int iW;
    
    /**
     * Height of given input. For example could be number of pixels in an image.
     */
    protected int iH;

    /**
     * Pooling width, i.e., number of individual inputs that will be averaged.
     */
    protected int pW;
    
    /**
     * Pooling height, i.e., number of individual inputs that will be averaged.
     */
    protected int pH;

    /**
     * Width stride, that is the number of inputs to move in width to next rectangle
     * that will be pooled. Typically sW = pW.
     */
    protected int sW;
    
    /**
     * Height stride, that is the number of inputs to move in width to next rectangle
     * that will be pooled. Typically sH = pH.
     */
    protected int sH;
    
    /**
     * Output height after mean pooling. Derived from iH, pH and sH.
     */
    protected int oH;

    /**
     * Output width after mean pooling. Derived from iH, pH and sH.
     */
    protected int oW;

    /**
     * Create new mean pooling layer assumes stride height = pH, and stride
     * width = pW.
     * @param pH Pooling height.
     * @param pW Pooling width.
     */
    public MeanPooling2DLayer(int pH, int pW) {
        this.pH = pH;
        this.pW = pW;
        this.sH = pH;
        this.sW = pW;
    }

    /**
     * Create new mean pooling layer with given inputs.
     * @param pH Pooling height.
     * @param pW Pooling width.
     * @param sH Stride height.
     * @param sW Stride width.
     */
    public MeanPooling2DLayer(int pH, int pW, int sH, int sW) {
        this.pH = pH;
        this.pW = pW;
        this.sH = sH;
        this.sW = sW;
    }

    /**
     * Set the dimensions of inputs not counting examples.
     * @param inputs Dimensions of inputs, e.g., 2,4,5. Assumes first 2 dimensions
     * are the input width and input height. If given a 3rd number assumes this
     * is the number of channels, else number of channels is set to 1.
     */
    @Override
    public void setDimsOfInputsWOE(int... inputs) {
        if(inputs.length != 3 && inputs.length != 2){
            LOG.error("Expected 3 or 2 inputs: channel(o), input width, input height");
            return;
        }
        iW = inputs[0];
        iH = inputs[1];
        
        if((iW - Math.max(pW, sW)) % sW != 0){
            LOG.error("The specified pooling width and stride does not allow to subsample correctly");
            return;
        }

        if((iH - Math.max(pH, sH)) % sH != 0){
            LOG.error("The specified pooling height and stride does not allow to subsample correctly");
            return;
        }
        
        oW = (iW - Math.max(pW,sW)) / sW + 1;
        oH = (iH - Math.max(pH,sH)) / sH + 1;
        
        if(inputs.length == 2){
            nI = new int[]{iW,iH,1};
            nC = 1;
        } else{
            nI = inputs;
            nC = inputs[2];
        }
    }

    /**
     * Return the dimension of output without examples.
     * @return Output Width, Output Height and number of channels.
     */
    @Override
    public int[] getDimsOfOutputsWOE() {
        return new int[]{oW,oH,nC};
    }

    /**
     * Create output and grad tensors, with input and output dimension + number
     * of examples.
     * @param numExamples Number of examples.
     */
    @Override
    public void initSpaceInMemory(int numExamples){
        if(output == null){
            output = new DoubleTensor(oW, oH, nC, numExamples);
            output.setPosToLast();
            grad = new DoubleTensor(iW, iH, nC, numExamples);
            grad.setPosToLast();
        } else{ // if tensors already exists just resize.
            output.resize(oW, oH, nC, numExamples);
            output.setPos(oW * oH * nC * numExamples);
            grad.resize(iW, iH, nC, numExamples);
            grad.setPos(iW * iH * nC * numExamples);
        }
    }

    /**
     * Forward propagate by pooling in 2D. Pooling in 2D entails obtaining the mean
     * for every example, every channel and every rectangle of height pH and 
     * width pW. Then move to next rectangle by stride width and height specified.
     * @param input Input Tensor.
     * @param isTest Whether test or training data (not used).
     * @return DoubleTensor result of forward propagation.
     */
    @Override
    public DoubleTensor fProp(DoubleTensor input, boolean isTest) {
        this.input = input;

        if(input.dims.length == 3){
            int[] ndims = new int[4];
            ndims[0] = input.dims[0];
            ndims[1] = input.dims[1];
            ndims[2] = 1;
            ndims[3] = input.dims[2];

            input.dims = ndims;
        }

        // pool2D
        return pool2D(
            input, sW, sH,
            output, nC, oW, oH,
            pW, pH, input.lastDim());
    }

    /**
     * Backpropagate the MeanPooling layer by calculating the derivative of the
     * loss function with respect to inputs. Since the derivative of the input
     * with respect to output = 1 / p2 (where p2 = pW * pH) by applying the chain
     * rule: add chaingrad_kl / p2 to every grad_ij (grad = output chaingrad)
     * such that output_kl = input_ij /p2 + ...
     * @param chainGrad Loss function with respect to outputs of meanpooling layer.
     * @return Loss function with respect to inputs of Meanpooling layer.
     */
    @Override
    public DoubleTensor bProp(DoubleTensor chainGrad) {
        // unpool2D
        return unpool2D(
            chainGrad, sW, sH,
            grad, nC, oW, oH,
            pW, pH, input.lastDim());
    }

    /**
     * Pool 2D: for every example, every channel and every rectangle of height
     * pH and width pW obtain mean, then move by sW and sH.
     * @param in DoubleTensor which to pool.
     * @param sW Stride width.
     * @param sH Stride height.
     * @param out Output DoubleTensor result of pooling.
     * @param nC Number of channels.
     * @param oW Output width.
     * @param oH Output height.
     * @param pW Pooling width.
     * @param pH Pooling height.
     * @param examples Number of examples.
     * @return Output tensor (same as out).
     */
    private DoubleTensor pool2D(
            DoubleTensor in, int sW, int sH,
            DoubleTensor out, int nC, int oW, int oH,
            int pW, int pH, int examples
            ){
        int[] indInput;
        int iIn, iO = 0;
        // iW,iH,c,e
        indInput = new int[4];
        // oW,oH,c,e
        int p2 = pW * pH;

        // for every example
        for(int e = 0; e < examples; e++){
            indInput[3] = e;
            // for every channel
            for(int c = 0; c < nC; c++){
                indInput[2] = c;
                for(int i = 0; i < oH; i++){
                    for(int j = 0; j < oW; j++){
                        // initialize to zero this value of output to accumulate
                        out.setQuick(iO, 0.0);
                        for(int k = 0; k < pH; k++){
                            indInput[1] = i * sH + k;
                            for(int l = 0; l < pW; l++){
                                indInput[0] = j * sW + l;
                                iIn = DoubleTensor.indicesToNum(indInput, in.dims);
                                // accumulate in output
                                out.setQuick(iO,
                                   out.getQuick(iO) + in.getQuick(iIn));
                            }
                        }
                        // Obtain mean after summing all entries.
                        out.setQuick(iO, out.getQuick(iO) / p2);

                        iO++;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Add chaingrad_kl / p2 to every grad_ij such that output_kl = input_ij /p2 + ...
     * @param chainGrad Incoming chaingrad.
     * @param sW Stride width.
     * @param sH Stride height.
     * @param grad Result chaingrad.
     * @param nC Number of channels.
     * @param oW Output width.
     * @param oH Output height.
     * @param pW Pooling width.
     * @param pH Pooling height.
     * @param examples Number of examples.
     * @return Result chaingrad.
     */
    private DoubleTensor unpool2D(
            DoubleTensor chainGrad, int sW, int sH,
            DoubleTensor grad, int nC, int oW, int oH,
            int pW, int pH, int examples
            ){
        int[] indGrad;
        int iCG = 0, iG;
        // oW,oH,c,e
        // iW,iH,c,e
        indGrad = new int[4];
        double p2 = pW * pH;

        // for every example
        for(int e = 0; e < examples; e++){
            indGrad[3] = e;
            // for every channel
            for(int c = 0; c < nC; c++){
                indGrad[2] = c;
                for(int i = 0; i < oH; i++){
                    for(int j = 0; j < oW; j++){
                        for(int k = 0; k < pH; k++){
                            indGrad[1] = i * sH + k;
                            for(int l = 0; l < pW; l++){
                                indGrad[0] = j * sW + l;
                                iG = DoubleTensor.indicesToNum(indGrad, grad.dims);
                                //output.put(i,j,act);
                                grad.setQuick(iG, chainGrad.getQuick(iCG) / p2);
                            }
                        }
                        iCG++;
                    }
                }
            }
        }
        return grad;
    }
}