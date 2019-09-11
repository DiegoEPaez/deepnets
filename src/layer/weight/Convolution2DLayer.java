package layer.weight;

import init.Initialization;
import tensor.DoubleTensor;
import org.apache.log4j.Logger;

/**
 *
 * @author diego_paez
 */
public class Convolution2DLayer extends WeightLayer {

    /*
     * Different implementations of 2D Conv.
     * CONV2D - brute force convolution ... very slow
     * IM2COL - every image patch is transformed into a row and matrix multiplied by filters
     *      which are organized as columns. see: High Performance Convolutional Neural Networks for
     *      Document Processing, Kumar Chellapilla
     * WINOGRAD - uses winograd filter, see: Fast Algorithms for Convolutional Neural Networks,
     *      Andrew Levin
     * FFT - use fast fourier transform, see: Fast Training of Convolutional Networks through
     *      FFTs, Michael Mathieu
     *  Also see Very Efficient Training of Convolutional Neural Networks using FFT
     *      Tyler Highlander
     *
     * See https://github.com/Maratyszcza/NNPACK, for winograd and fft fast implementations
     * The fastest seems to be FFT (as implemented in NNPACK, see comparison)
     */
    public enum ConvolveMethod{
        CONV2D,
        IM2COL,
        WINOGRAD,
        FFT;
    }

    private static final Logger log = Logger.getLogger(Convolution2DLayer.class);

    // number of channels
    protected int nC;

    // number of kernels
    protected int nK;

    // input width
    protected int iW;

    // input height
    protected int iH;

    // kernel width
    protected int kW;

    // kernel height
    protected int kH;

    // output width
    protected int oW;
    
    // output height
    protected int oH;

    // stride
    protected int sW;
    protected int sH;

    private ConvFFT cfft;
    protected ConvolveMethod method;


    public Convolution2DLayer(Initialization init, int numFilters, int kH, int kW) {
        this.init = init;
        this.nK = numFilters;
        this.kH = kH;
        this.kW = kW;
        this.sH = 1;
        this.sW = 1;
        this.method = ConvolveMethod.FFT;
    }

    public Convolution2DLayer(Initialization init, int numFilters, int kH, int kW,
            int sH,int sW) {
        this.init = init;
        this.nK = numFilters;
        this.kW = kW;
        this.kH = kH;
        this.sW = sW;
        this.sH = sH;
        this.method = ConvolveMethod.FFT;
    }

    public Convolution2DLayer(Initialization init, int numFilters, int kW, int kH,
            int sW, int sH, ConvolveMethod method) {
        this.init = init;
        this.nK = numFilters;
        this.kW = kW;
        this.kH = kH;
        this.sW = sW;
        this.sH = sH;
        this.method = method;
    }

    @Override
    public void setDimsOfInputsWOE(int... inputs) {
        if(inputs.length != 3 && inputs.length != 2){
            log.error("Expected 3 or 2 inputs: channel(o), input width, input height");
            return;
        }

        /* assumes width first, i.e., assumes row-major order, although,
         * DoubleMatrix from jblas uses column-major order, libraries like
         * fftw or jtransforms use row-major order and thus to keep consistency
         * with these libraries row-major order is assumed for convolutions.
         */
        iW = inputs[0];
        iH = inputs[1];

        if((iW - Math.max(kW, sW)) % sW != 0){
            log.error("The specified stride does not allow to cross correlate the width correctly");
            return;
        }
        if((iH - Math.max(kH, sH)) % sH != 0){
            log.error("The specified stride does not allow to cross correlate the height correctly");
            return;
        }

        oW = (iW - Math.max(kW,sW)) / sW + 1;
        oH = (iH - Math.max(kH,sH)) / sH + 1;
        
        if(inputs.length == 2){
            nI = new int[]{iW,iH,1};
            nC = 1;
        } else{
            nI = inputs;
            nC = inputs[2];
        }
    }

    @Override
    public int[] getDimsOfOutputsWOE() {
        return new int[]{oW, oH, nK};
    }

    @Override
    public void initParams() {
        bias = new DoubleTensor(init.initBias(nK), nK);
        biasGrad = new DoubleTensor(bias.dims);
        biasGrad.setPosToLast();

        weights = new DoubleTensor(init.initWeights(nC, nK, new int[]{kW, kH, nC, nK}),
                new int[]{kW, kH, nC, nK});
        weightsGrad = new DoubleTensor(weights.dims);
        weightsGrad.setPosToLast();
    }

    @Override
    public void initSpaceInMemory(int numExamples){
        if(output == null){
            output = new DoubleTensor(oW, oH, nK, numExamples);
            output.setPosToLast();
            grad = new DoubleTensor(iW, iH, nC, numExamples);
            grad.setPosToLast();
        } else{
            output.resize(oW, oH, nK, numExamples);
            output.setPos(oW * oH * nK * numExamples);
            grad.resize(iW, iH, nC, numExamples);
            grad.setPos(iW * iH * nC * numExamples);
        }
    }

    @Override
    public int getNumberOfParams() {
        return (kW * kH * nC + 1) * nK;
    }

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
        if(method == ConvolveMethod.CONV2D){
            // input size = nC, iH, iW, examples
            DoubleTensor res = ConvBruteForce.fwdFilter(
                input, iW, iH, sW, sH,
                weights, nC, kW, kH,
                output, nK, oW, oH,
                bias,
                input.lastDim());
            return res;
        } else { // FFT
            if(cfft == null)
                cfft = new ConvFFT();
            cfft.fwdFilter(input, iW, iH, sW, sH, weights, nC, kW, kH,
                    output, nK, oW, oH, bias, input.lastDim());
            return output;
        }
    }

    @Override
    public DoubleTensor bProp(DoubleTensor chainGrad) {
        if(method == ConvolveMethod.CONV2D){
            // chainGrad size = nK, oH, oW, examples
            grad = ConvBruteForce.bwdFilter(
                chainGrad, sW, sH,
                weights, nK, kW, kH,
                grad, nC, iW, iH,
                input.lastDim());
            return grad;
        } else { // FFT
            cfft.bwdFilter(chainGrad, sH, sW, oH, oW, weights, nK, kH, kW,
                    grad, nC, iH, iW, input.lastDim());
            return grad;
        }
    }

    @Override
    public void updateLayerWGrad(DoubleTensor chainGrad) {
        if(method == ConvolveMethod.CONV2D){
            // input size = nC, iH, iW, examples

            // chainGrad = oW, oH, nK, e
            // weights =  kW,kH,c,f
            ConvBruteForce.weightUpdate(
                input, sH, sW,
                chainGrad, nK, oW, oH,
                weightsGrad, nC, kW, kH,
                input.lastDim());

            // chainGrad size = nK, oW, oH, examples - aggregate by num filters
            chainGrad.sumAlli(2, biasGrad);
        } else{ // FFT
            /*cfft.gradConvolve(chainGrad, sH, sW, oH, oW, weights, nK, kH, kW,
                    grad, nC, iH, iW, input.lastDim());*/
            // input size = nC, iH, iW, examples

            // chainGrad = oW, oH, nK, e
            // weights =  kW,kH,c,f
            // needs specialized method due to channels && stuff
            cfft.weightUpdate(
                input, sH, sW, kW, kH,
                chainGrad, nK, oW, oH,
                weightsGrad, nC, iW, iH,
                input.lastDim());

            // chainGrad size = nK, oW, oH, examples - aggregate by num filters
            chainGrad.sumAlli(2, biasGrad);
        }
    }
}