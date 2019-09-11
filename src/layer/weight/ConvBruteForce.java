package layer.weight;

import tensor.DoubleTensor;

/**
 * The methods necessary to forward propagate a Convolutional Layer (currently
 * only 2D) and back propagate it using brute force are defined in this method.
 * Brute force means applying convolutions by their definition, iterating over
 * every element to multiply it as many times as necessary. This contrasts with
 * other methods like: fast fourier transform, image to column and winograd filters.
 * @author diego_paez
 */
public class ConvBruteForce {


    /**
     * Applies a cross correlation (very similar to a convolution except data
     * is not flipped), the cross correlation is only over the parts where both
     * matrices completely overlap. The cross correlation is between input tensor
     * and kernel tensor, and is used to forward propagate a convolutional 2D 
     * layer. Uses supplied out Tensor to save data.
     * Besides cross correlating, the bias is also added (or rather output element
     * is initialized with bias value).
     *
     * @param in DoubleTensor which has input to which a kernel will be applied
     * to cross correlate.
     * @param iW Input Width.
     * @param iH Input Height.
     * @param sW Stride Width.
     * @param sH Stride Height.
     * @param kernel Double Tensor that contains kernels which will be applied to
     * Input tensor (weights of convolutional layer).
     * @param nC Number of channels.
     * @param kW Kernel Width.
     * @param kH Kernel Height.
     * @param out Output Tensor.
     * @param nK Number of Kernels
     * @param oW Output width.
     * @param oH Output height.
     * @param bias Bias that will be added to cross correlated result.
     * @param eN Number of examples.
     * @return Output tensor.
     *
     */
    public static DoubleTensor fwdFilter(
            DoubleTensor in, int iW, int iH, int sW, int sH,
            DoubleTensor kernel, int nC, int kW, int kH,
            DoubleTensor out, int nK, int oW, int oH,
            DoubleTensor bias,
            int eN){
        int[] indInput, indFilter;
        int iIn, iF, iO = 0;
        // Input Dims: iW,iH,nC,e
        indInput = new int[4];
        // Kernel Tensor dims: kW,kH,nC,nK
        indFilter = new int[4];
        // Output dims: oW,oH,nK,e

        // for every example
        for(int e = 0; e < eN; e++){
            indInput[3] = e;
            // for every filter
            for(int f = 0; f < nK; f++){
                indFilter[3] = f;
                for(int i = 0; i < oH; i++){
                    for(int j = 0; j < oW; j++){
                        // Set bias if different to null as initial value
                        if(bias != null)
                            out.setQuick(iO, bias.getQuick(f));
                        else
                            out.setQuick(iO, 0.0);

                        // for every channel crosscorrelate in 2d, adding all up
                        for(int c = 0; c < nC; c++){
                            indFilter[2] = c;
                            indInput[2] = c;
                            for(int k = 0; k < kH; k++){
                                indFilter[1] = k;
                                indInput[1] = i * sH + k;
                                for(int l = 0; l < kW; l++){
                                    indFilter[0] = l;
                                    indInput[0] = j * sW + l;
                                    iIn = DoubleTensor.indicesToNum(indInput, in.dims);
                                    iF = DoubleTensor.indicesToNum(indFilter, kernel.dims);
                                    //output.put(i,j,act);
                                    out.data.addQuick(iO,
                                            in.getQuick(iIn) * kernel.getQuick(iF));
                                }
                            }
                        }
                        iO++;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Applies a full convolution 2d, the convolution includes the borders where
     * the filter does not completely overlap the matrix. The convolution is
     * between the incoming chainGrad  tensor and the kernel (weights) of the
     * convolutional layer. The result of the convolution is the derivative of
     * the loss function with respect to the inputs of the convolutional layer.
     *
     * @param chainGrad Incoming chainGrad.
     * @param sH Stride Height.
     * @param sW Stride Width.
     * @param kernel Tensor with kernels.
     * @param nK Number of kernels.
     * @param kH Kernel Height.
     * @param kW Kernel Width.
     * @param grad Derivative of loss function w.r. to inputs, which
     * will be calculated by this method.
     * @param nC Number of Channels.
     * @param iH Input Height.
     * @param iW Input Width.
     * @param examples Number of Examples.
     * @return Grad = output chainGrad.
     */
    public static DoubleTensor bwdFilter(
            DoubleTensor chainGrad, int sH, int sW,
            DoubleTensor kernel, int nK, int kH, int kW,
            DoubleTensor grad, int nC, int iH, int iW,
            int examples){
        int bigBy1, bigBy2, start1, end1, start2, end2;
       
        bigBy1 = iH - kH;
        bigBy2 = iW - kW;

        int[] indInput, indFilter;
        int iIn, iF, iO = 0;
        // oH,oW,f,e
        indInput = new int[4];
        // kH,kW,c,f
        indFilter = new int[4];
        // iH,iW,c,e

        for(int e = 0; e < examples; e++){
            indInput[3] = e;
            // for every channel
            for(int c = 0; c < nC; c++){
                indFilter[2] = c;
                for(int i = 0; i < iH; i++){

                    /* find out start and end of loop: in the middle start = filter.length -1 end = 0
                     * ,i.e., loop through the whole filter. But in the borders of convolution loop is
                     * shorter thus start and end index ought to be calculated.
                     */
                    start1 = Math.min(kH - 1,i);
                    end1 = Math.max(0, i - bigBy1);

                    for(int j = 0; j < iW; j++){

                        grad.setQuick(iO, 0.0);

                        start2 = Math.min(kW - 1,j);
                        end2 = Math.max(0, j - bigBy2);

                        for(int f = 0; f < nK; f++){
                            indInput[2] = f;
                            indFilter[3] = f;
                            // set bias  value
                            for(int k = start1; k >= end1; k--){
                                indFilter[1] = k;
                                indInput[1] = i * sH - k;

                                for(int l = start2; l >= end2; l--){
                                    indInput[0] = j * sW - l;
                                    indFilter[0] = l;
                                    iIn = DoubleTensor.indicesToNum(indInput, chainGrad.dims);
                                    iF = DoubleTensor.indicesToNum(indFilter, kernel.dims);

                                    grad.data.addQuick(iO,
                                            chainGrad.getQuick(iIn) * kernel.getQuick(iF));
                                }
                            }
                        }
                        iO++;
                    }
                }
            }
        }
        return grad;
    }
    
    /**
     * Applies a cross correlation (very similar to a convolution except data
     * is not flipped), the cross correlation is only over the parts where both
     * matrices completely overlap. The cross correlation is between the input
     * to a convolutional 2D layer and the incoming chainGrad from the next 
     * layer. By cross correlating the gradient with respect to theta can be
     * obtained to update the parameters of the convolutional net.
     * 
     * @param in Input Tensor of to cross correlate (input of convolutional layer).
     * @param sH Stride Height.
     * @param sW Stride Width.
     * @param chainGrad Incoming chainGrad from next layer that will be 
     * cross correlated with input.
     * @param nC Number of Channels.
     * @param kH Kernel Height.
     * @param kW Kernel Width.
     * @param gradWeights Gradient of weights, where the result of the cross
     * correlation will be stored.
     * @param nK Number of Kernels.
     * @param oH Output Height.
     * @param oW Output Width.
     * @param examples Number of examples.
     * @return Gradient of weights (not including gradient of bias).
     */
    public static DoubleTensor weightUpdate(
            DoubleTensor in, int sH, int sW,
            DoubleTensor chainGrad, int nK, int oH, int oW,
            DoubleTensor gradWeights, int nC, int kH, int kW,
            int examples){
        int[] indInput, indChainGrad;
        int iIn, iF, iO = 0;
        // Dimensions of Input Tensor: iW,iH,nC,e
        indInput = new int[4];
        // Dimensions of ChainGrad: oW,oH,nK,e
        indChainGrad = new int[4];
        // Dimensions of gradient: kW,kH,nC,nK

        for(int f = 0; f < nK; f++){
            indChainGrad[2] = f;
            for(int c = 0; c < nC; c++){
                indInput[2] = c;
                for(int k = 0; k < kH; k++){
                    for(int l = 0; l < kW; l++){

                        indChainGrad[0] = f;
                        // init to zero.
                        gradWeights.setQuick(iO, 0.0);
                        for(int e = 0; e < examples; e++){
                            indInput[3] = e;
                            indChainGrad[3] = e;
                            for(int i = 0; i < oH; i++){
                                indChainGrad[1] = i;
                                indInput[1] = i * sH + k;
                                for(int j = 0; j < oW; j++){
                                    indChainGrad[0] = j;
                                    indInput[0] = j * sW + l;
                                    iF = DoubleTensor.indicesToNum(indChainGrad, chainGrad.dims);
                                    iIn = DoubleTensor.indicesToNum(indInput, in.dims);
                                    //output.put(i,j,act);
                                    gradWeights.data.addQuick(iO,
                                            in.getQuick(iIn) * chainGrad.getQuick(iF));
                                }
                            }
                        }
                        iO++;
                    }
                }
            }
        }
        return gradWeights;
    }
}
