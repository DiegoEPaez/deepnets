package layer.weight;

import org.jtransforms.fft.DoubleFFT_2D;
import tensor.DoubleTensor;
import org.apache.log4j.Logger;

/**
 * Uses FFT to do faster convolutions. A speed up might be achieved by using
 * vectorized instructions (SIMD).
 * @author diego_paez
 */
public class ConvFFT {

    /**
     * DoubleFFT_2D from jtransforms that is used to do fft transformations.
     */
    private DoubleFFT_2D kfft;
    
    /**
     * Stores fft of Kernel. Dimensions are: 2, nK, nC, kfH, kfW.
     */
    private DoubleTensor fftKernel;
    
    /**
     * Stores fft of Input. Dimensions: 2, ex, nC, overlapW, overlapH, kfH,kfW.
     * (might have many kernels, as many as overlaps there may be)
     * overlapW = ifW / kfW
     */
    private DoubleTensor fftInput;
    
    /**
     * Stores fft of Input. Dimensions:2, ofW, ofH, overlapLW, overlapLH, nC, ex.
     * Note overlap dimensions are different thus, a different DoubleTensor is used
     * overlapLW = ifW / ofW
     */
    private DoubleTensor fftInput2;
    
    /**
     * Stores result of the complex multiplication of fftInput and fftKernel
     * in fwdFilter Method
     * Dimensions: 2, kfW, kfH, overlapW, overlapH, nK, ex.
     */
    private DoubleTensor fftMult;
    
    /**
     * Stores result of the complex multiplication of fftChainGrad and fftKernel.
     * Dimensions: 2, kfW, kfH, overlapOW, overlapOH, nC, ex.
     */
    private DoubleTensor fftMult2;
    
    /**
     * Stores result of the complex multiplication of fftInput2 and fftChainGrad2.
     * Dimensions: 2, ofW, ofH, overlapLW, overlapLH, nC, nK.
     * 
     */
    private DoubleTensor fftMult3;
    
    /**
     * Stores fft of ChainGrad.
     * Dimensions: 2, kfW, kfH, overlapOW, overlapOH, nK, ex.
     */
    private DoubleTensor fftChainGrad;
    
    /**
     * Stores fft of ChainGrad.
     * Dimensions: 2, ofW, ofH, nK, ex.
     */
    private DoubleTensor fftChainGrad2;

    /**
     * Log4j logger.
     */
    private static final Logger LOG = Logger.getLogger(ConvFFT.class);

    /**
     * Forward propagate a 2D convolutional layer using fft.
     * Currently stride is not used.
     * 
     * @param in DoubleTensor which has input to which a kernel will be applied
     * to cross correlate.
     * @param iW Input Width.
     * @param iH Input Height.
     * @param sW Stride Width.
     * @param sH Stride Height.
     * @param kernel Double Tensor that cointans kernels which will be applied to
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
     */
    public DoubleTensor fwdFilter(
            DoubleTensor in, int iW, int iH, int sW, int sH,
            DoubleTensor kernel, int nC, int kW, int kH,
            DoubleTensor out, int nK, int oW, int oH,
            DoubleTensor bias,
            int eN) {
        /* Calculate dimensions to apply fft since fft can only be applied to
        powers of 2 move to next power and then multiply by 2, due to the need
        of having complex numbers as well as real numbers.
        */
        int kfH = nextPowerOf2(kH) << 1;
        int kfW = nextPowerOf2(kW) << 1;
        int ifH = nextPowerOf2(iH) << 1;
        int ifW = nextPowerOf2(iW) << 1;
        
        // Calculate overlap, i.e., how many times each kernel fits in input
        int overlapW = ifW / kfW;
        int overlapH = ifH / kfH;
        int ex = in.lastDim();

        if (kfft == null) {
            kfft = new DoubleFFT_2D(kfW, kfH);
        }

        // kW,kH,c,f - store fft of Kernel
        if(fftKernel == null){
            fftKernel = new DoubleTensor(2, kfW, kfH, nC, nK);
            fftKernel.setPosToLast();
        }
        // iW,iH,c,e - store fft of Input
        if(fftInput == null){
            fftInput = new DoubleTensor(2, kfW, kfH, overlapW, overlapH, nC, ex);
            fftInput.setPosToLast();
        }
        if(fftMult == null){
            fftMult = new DoubleTensor(2, kfW, kfH, overlapW, overlapH, nK, ex);
            fftMult.setPosToLast();
        }

        LOG.debug("fftKernel");
        kernel2DFT(kernel, fftKernel, nK, nC, kH, kW, true);
        LOG.debug("fftInput");
        input2DFT(in, fftInput, in.lastDim(), nC, overlapW, overlapH, kW, kH, iW, iH);
        LOG.debug("fftMult");
        multiply(fftInput, fftKernel, fftMult);
        LOG.debug("output2IDFT");
        output2IDFT(kH, kW, eN, nK, overlapW, overlapH, oW, oH, fftMult, bias, out, true);


        // free space fftInput and fftMult wont be used again (till next pass)
        fftInput = null;
        fftMult = null;
        return out;
    }

    /**
     * 
     * @param chainGrad
     * @param sH
     * @param sW
     * @param oH
     * @param oW
     * @param kernel
     * @param nK
     * @param kH
     * @param kW
     * @param grad
     * @param nC
     * @param iH
     * @param iW
     * @param ex
     * @return 
     */
    public DoubleTensor bwdFilter(
            DoubleTensor chainGrad, int sH, int sW, int oH, int oW,
            DoubleTensor kernel, int nK, int kH, int kW,
            DoubleTensor grad, int nC, int iH, int iW,
            int ex){
        int kfH = nextPowerOf2(kH) << 1;
        int kfW = nextPowerOf2(kW) << 1;
        int ofH = nextPowerOf2(oH) << 1;
        int ofW = nextPowerOf2(oW) << 1;
        int overlapOW = ofW / kfW;
        int overlapOH = ofH / kfH;

        kernel2DFT(kernel, fftKernel, nK, nC, kH, kW, false);

        if(fftChainGrad == null){
            fftChainGrad = new DoubleTensor(2, kfW, kfH, overlapOW, overlapOH, nK, ex);
            fftChainGrad.setPosToLast();
        }
        if(fftMult2 == null){
            fftMult2 = new DoubleTensor(2, kfW, kfH, overlapOW, overlapOH, nC, ex);
            fftMult2.setPosToLast();
        }

        // Calculate fft of chainGrad oH,oW,f,e
        input2DFT(chainGrad, fftChainGrad, ex, nK, overlapOW, overlapOH, kW, kH, oW, oH);
        multiply2(fftChainGrad, fftKernel,fftMult2);

        // out = grad - iW, iH, nC, numExamples
        output2IDFT(kH, kW, ex, nC, overlapOW, overlapOH, iW, iH, fftMult2, null, grad, false);

        // free space
        fftChainGrad = null;
        fftMult2 = null;
        return grad;
    }

    /**
     * 
     * @param in
     * @param sH
     * @param sW
     * @param kH
     * @param kW
     * @param chainGrad
     * @param nK
     * @param oH
     * @param oW
     * @param gradWeights
     * @param nC
     * @param iH
     * @param iW
     * @param ex
     * @return 
     */
    public DoubleTensor weightUpdate(
            DoubleTensor in, int sH, int sW, int kH, int kW,
            DoubleTensor chainGrad, int nK, int oH, int oW,
            DoubleTensor gradWeights, int nC, int iH, int iW,
            int ex){
        int ofH = nextPowerOf2(oH) << 1;
        int ofW = nextPowerOf2(oW) << 1;
        int ifH = nextPowerOf2(iH) << 1;
        int ifW = nextPowerOf2(iW) << 1;
        int overlapLW = ifW / ofW;
        int overlapLH = ifH / ofH;

        if(fftInput2 == null){
            fftInput2 = new DoubleTensor(2, ofW, ofH, overlapLW, overlapLH, nC, ex);
            fftInput2.setPosToLast();
        }
        if(fftChainGrad2 == null){
            fftChainGrad2 = new DoubleTensor(2, ofW, ofH, nK, ex);
            fftChainGrad2.setPosToLast();
        }
        if(fftMult3 == null){
            fftMult3 = new DoubleTensor(2, ofW, ofH, overlapLW, overlapLH, nC, nK);
            fftMult3.setPosToLast();
        }

        input2DFT(in, fftInput2, in.lastDim(), nC, overlapLW, overlapLH, oW, oH, iW, iH);

        // chainGrad = oW, oH, nK, e
        kernel2DFT(chainGrad, fftChainGrad2, ex, nK, oW, oH, true);
        multiply3(fftInput2, fftChainGrad2, fftMult3);

        output2IDFT(oH, oW, nK, nC, overlapLW, overlapLH, kW, kH, fftMult3, null, gradWeights, true);

        // free space
        fftInput2 = null;
        fftChainGrad2 = null;
        fftMult3 = null;
        return gradWeights;
    }

    /**
     * 
     * @param kernel
     * @param output
     * @param nK
     * @param nC
     * @param kW
     * @param kH
     * @param flip 
     */
    private void kernel2DFT(DoubleTensor kernel, DoubleTensor output, int nK, int nC,
            int kW, int kH, boolean flip) {
        // flip = true to cross correlate
        int kfH = nextPowerOf2(kH) << 1;
        int kfW = nextPowerOf2(kW) << 1;

        double[] kernelTemp = new double[kfW * kfH * 2];
        int[] kernelInd = new int[4];
        int[] kernelDim = new int[]{kfW, kfH, nC, nK};
        int indFft = 0;

        // First compute all FFT's of kernels and store them. Note that
        // we only need to compute them once, not once per example
        int kerI = 0, wInI = 0;
        for (int k = 0; k < nK; k++) {
            kernelInd[3] = k;
            for (int c = 0; c < nC; c++) {
                kernelInd[2] = c;
                
                // create matrix for computing dft of each filter
                if(!flip){
                    kerI = 0;
                    for (int i = 0; i < kfH; i++) {
                        for (int j = 0; j < kfW; j++) {
                            if(i < kH && j < kW){
                                kernelTemp[kerI++] = kernel.getQuick(wInI);
                                wInI++;
                                kernelTemp[kerI++] = 0;
                            } else{
                                kernelTemp[kerI++] = 0;
                                kernelTemp[kerI++] = 0;
                            }
                        }
                    }
                } else{
                    kerI = 2 * kfH * kfW - 1;
                    for (int i = kfH - 1; i >= 0; i--) {
                        for (int j = kfW - 1; j >= 0; j--) {
                            if(i < kH && j < kW){
                                kernelTemp[kerI--] = 0;
                                kernelTemp[kerI--] = kernel.getQuick(wInI);
                                wInI++;
                            } else{
                                kernelTemp[kerI--] = 0;
                                kernelTemp[kerI--] = 0;
                            }
                        }
                    }
                }
                // compute dft of kernel
                kfft.complexForward(kernelTemp);

                // copy dft of kernel into tensor, data change positions to perform cmmul
                kerI = 0;
                for (int i = 0; i < kfH; i++) {
                    kernelInd[1] = i;
                    for (int j = 0; j < kfW; j++) {
                        kernelInd[0] = j;
                        indFft = DoubleTensor.indicesToNum(kernelInd, kernelDim);
                        indFft <<= 1;
                        output.data.setQuick(indFft++, kernelTemp[kerI++]);
                        output.data.setQuick(indFft, kernelTemp[kerI++]);
                    }
                }
            }
        }
    }

    /**
     * 
     * @param input
     * @param output
     * @param ex
     * @param nC
     * @param overlapW
     * @param overlapH
     * @param kW
     * @param kH
     * @param iW
     * @param iH 
     */
    private void input2DFT(DoubleTensor input, DoubleTensor output, int ex, int nC,
            int overlapW, int overlapH,
            int kW, int kH, int iW, int iH) {
        int k2H = nextPowerOf2(kH);
        int k2W = nextPowerOf2(kW);
        int kfH = k2H << 1;
        int kfW = k2W << 1;

        double[] inputTemp = new double[kfW * kfH * 2];
        int[] origInd = new int[4];
        int[] origDim = new int[]{iW, iH, nC, ex};
        int indFft = 0;

        // First compute all FFT's
        int indI = 0, wInI;
        for (int e = 0; e < ex; e++) {
            origInd[3] = e;
            for (int c = 0; c < nC; c++) {
                origInd[2] = c;
                for (int ovH = 0; ovH < overlapH; ovH++) {
                    for (int ovW = 0; ovW < overlapW; ovW++) {
                        indI = 0;
                        // create matrix for computing dft of each filter
                        for (int i = 0; i < kfH; i++) {
                            origInd[1] = ovH * k2H + i;
                            for (int j = 0; j < kfW; j++) {
                                if(i < k2H && j < k2W){
                                    origInd[0] = ovW * k2W + j;
                                    if(origInd[0] < iW  && origInd[1] < iH){
                                        wInI = DoubleTensor.indicesToNum(origInd, origDim);
                                        inputTemp[indI++] = input.getQuick(wInI);
                                        inputTemp[indI++] = 0;
                                    } else{
                                        inputTemp[indI++] = 0;
                                        inputTemp[indI++] = 0;
                                    }
                                } else{
                                    inputTemp[indI++] = 0;
                                    inputTemp[indI++] = 0;
                                }
                            }
                        }

                        kfft.complexForward(inputTemp);

                        // copy dft of input into tensor, data change positions to perform cmmul
                        indI = 0;
                        for (int i = 0; i < kfH; i++) {
                            for (int j = 0; j < kfW; j++) {
                                output.data.setQuick(indFft++, inputTemp[indI++]);
                                output.data.setQuick(indFft++, inputTemp[indI++]);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     *
     * @param matrix
     * @param filter
     * @param output
     */
    private void multiply(DoubleTensor matrix, DoubleTensor filter, DoubleTensor output) {
        // ker - 2, kfW, kfH, nC, nK
        // inp - 2, kfW, kfH, overlapW, overlapH, nC, ex
        // out - 2, kfW, kfH, overlapW, overlapH, nK, ex
        // fftKernel - 2, kfW, kfH, nC, nK
        // fftChainGrad - 2, kfW, kfH, overlapOW, overlapOH, nK, ex
        int kfW = matrix.dims[1];
        int kfH = matrix.dims[2];
        int overlapW = matrix.dims[3];
        int overlapH = matrix.dims[4];
        int ex = matrix.dims[6];
        int nC = filter.dims[3];
        int nK = filter.dims[4];
        
        int indM, indMCache, indOut, indOutCache, indK, indKCache1, indKCache2;
        double reI, reK;
        double imI, imK;
        double k1, k2, k3;
        indM = 0;
        indOut = 0;
        for (int e = 0; e < ex; e++) {
            indK = 0;
            indMCache = indM;
            for (int k = 0; k < nK; k++) {
                indM = indMCache;
                indOutCache = indOut;
                for (int c = 0; c < nC; c++) {
                    indOut = indOutCache;
                    indKCache1 = indK;
                    for (int ovH = 0; ovH < overlapH; ovH++) {
                        indK = indKCache1;
                        indKCache2 = indK;
                        for (int ovW = 0; ovW < overlapW; ovW++) {
                            indK = indKCache2;
                            for (int kH = 0; kH < kfH; kH++) {
                                for (int kW = 0; kW < kfW; kW++) {
                                    reI = matrix.data.getQuick(indM++);
                                    imI = matrix.data.getQuick(indM++);
                                    reK = filter.data.getQuick(indK++);
                                    imK = filter.data.getQuick(indK++);
                                    k1 = reI * (reK + imK);
                                    k2 = imK * (reI + imI);
                                    k3 = reK * (imI - reI);
                                    output.data.addQuick(indOut, k1 - k2);
                                    indOut++;
                                    output.data.addQuick(indOut, k1 + k3);
                                    indOut++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * 
     * @param chainGrad
     * @param filter
     * @param output 
     */
    private void multiply2(DoubleTensor chainGrad, DoubleTensor filter, DoubleTensor output) {
        // fftKernel - 2, kfW, kfH, nC, nK
        // fftChainGrad - 2, kfW, kfH, overlapOW, overlapOH, nK, ex
        // out - 2, kfW, kfH, overlapW, overlapH, nC, ex
        int kfW = chainGrad.dims[1];
        int kfH = chainGrad.dims[2];
        int overlapOW = chainGrad.dims[3];
        int overlapOH = chainGrad.dims[4];
        int ex = chainGrad.dims[6];
        int nC = filter.dims[3];
        int nK = filter.dims[4];

        int indM, indMCache, indOut, indOutCache, indK, indKCache1, indKCache2;
        double reI, reK;
        double imI, imK;
        double k1, k2, k3;
        indM = 0;
        indOut = 0;
        for (int e = 0; e < ex; e++) {
            indK = 0;
            indOutCache = indOut;
            for (int k = 0; k < nK; k++) {
                indOut = indOutCache;
                indMCache = indM;
                for (int c = 0; c < nC; c++) {
                    indM = indMCache;
                    indKCache1 = indK;
                    for (int ovH = 0; ovH < overlapOH; ovH++) {
                        indK = indKCache1;
                        indKCache2 = indK;
                        for (int ovW = 0; ovW < overlapOW; ovW++) {
                            indK = indKCache2;
                            for (int kH = 0; kH < kfH; kH++) {
                                for (int kW = 0; kW < kfW; kW++) {
                                    reI = chainGrad.data.getQuick(indM++);
                                    imI = chainGrad.data.getQuick(indM++);
                                    reK = filter.data.getQuick(indK++);
                                    imK = filter.data.getQuick(indK++);
                                    k1 = reI * (reK + imK);
                                    k2 = imK * (reI + imI);
                                    k3 = reK * (imI - reI);
                                    output.data.addQuick(indOut, k1 - k2);
                                    indOut++;
                                    output.data.addQuick(indOut, k1 + k3);
                                    indOut++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * 
     * @param input
     * @param chainGrad
     * @param output 
     */
    private void multiply3(DoubleTensor input, DoubleTensor chainGrad, DoubleTensor output) {
        // fftChainGrad - 2, kfW, kfH, nK, ex
        // fftInput - 2, kfW, kfH, overlapW, overlapH, nC, ex
        // out - 2, ofW, ofH, overlapLW, overlapLH, nC, nK
        int kfW = input.dims[1];
        int kfH = input.dims[2];
        int overlapLW = input.dims[3];
        int overlapLH = input.dims[4];
        int nC = input.dims[5];
        int ex = input.dims[6];
        int nK = chainGrad.dims[3];

        int indM, indMCache, indOut, indK, indKCache0, indKCache1, indKCache2;
        double reI, reK;
        double imI, imK;
        double k1, k2, k3;
        indM = 0;
        indK = 0;

        for (int e = 0; e < ex; e++) {
            indOut = 0;
            indMCache = indM;
            for (int k = 0; k < nK; k++) {
                indM = indMCache;
                indKCache0 = indK;
                for (int c = 0; c < nC; c++) {
                    indK = indKCache0;
                    indKCache1 = indK;
                    for (int ovH = 0; ovH < overlapLH; ovH++) {
                        indK = indKCache1;
                        indKCache2 = indK;
                        for (int ovW = 0; ovW < overlapLW; ovW++) {
                            indK = indKCache2;
                            for (int kH = 0; kH < kfH; kH++) {
                                for (int kW = 0; kW < kfW; kW++) {
                                    reI = input.data.getQuick(indM++);
                                    imI = input.data.getQuick(indM++);
                                    reK = chainGrad.data.getQuick(indK++);
                                    imK = chainGrad.data.getQuick(indK++);
                                    k1 = reI * (reK + imK);
                                    k2 = imK * (reI + imI);
                                    k3 = reK * (imI - reI);
                                    output.data.addQuick(indOut, k1 - k2);
                                    indOut++;
                                    output.data.addQuick(indOut, k1 + k3);
                                    indOut++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     *
     * @param nCK
     * @param kW
     * @param ex
     * @param nK
     * @param overlapW
     * @param overlapH
     * @param okW output size for the kernel sized convolution
     * @param okH
     * @param oW
     * @param oH
     */
    private void output2IDFT(int kH, int kW, int ex, int nCK,
            int overlapW, int overlapH, int oW, int oH,
            DoubleTensor fftMult, DoubleTensor bias, DoubleTensor output, boolean same){
        int k2H = nextPowerOf2(kH);
        int k2W = nextPowerOf2(kW);
        int kfH = k2H << 1;
        int kfW = k2W << 1;

        // kW,kH,c,f - store fft of Kernel
        int patchSize = kfW * kfH * 2;
        int cH, cW;
        cH = kH + k2H - 1;
        cW = kW + k2W - 1;
        
        double[] outTemp = new double[patchSize];

        // 2, kfW, kfH, overlapW, overlapH, nK, ex
        int pos = 0;
        int ind = 0;
        int[] indA = new int[4];
        int[] pInd = new int[2];
        int[] dimsP = new int[]{kfW,kfH};
        int[] dims = new int[]{oW, oH, nCK, ex};
        int pI = 0;

        // init with bias
        double biasF;
        for(int e = 0; e < ex; e++){
            for(int f = 0; f < nCK; f++){
                if(bias != null)
                    biasF = bias.data.getQuick(f);
                else
                    biasF = 0.0;
                for(int i = 0; i < oH; i++){
                    for(int j = 0; j < oW; j++){
                        output.data.setQuick(ind, biasF);
                        ind++;
                    }
                }
            }
        }

        ind = 0;
        for (int e = 0; e < ex; e++) {
            indA[3] = e;
            for(int f = 0; f < nCK; f++){
                indA[2] = f;
                for(int ovH = 0; ovH < overlapH; ovH++){
                    for(int ovW = 0; ovW < overlapW; ovW++){
                        System.arraycopy(fftMult.data.getData(), pos, outTemp,
                                0, patchSize);
                        kfft.complexInverse(outTemp, true);
                        pos += patchSize;

                        pI = 0;
                        for(int i = 0; i < cH; i++){
                            if(same == true){
                                if(ovH == 0 && i < kH - 1){
                                    continue;
                                }
                                if(ovH == overlapH - 1 && i > cH - kH){
                                    continue;
                                }

                                indA[1] = ovH * k2H + i - (kH - 1);
                            } else {
                                indA[1] = ovH * k2H + i;
                            }
                            // continue when the input is not a multiple of 2
                            if(indA[1] >= oH)
                                continue;

                            pInd[1] = i;
                            for(int j = 0; j < cW; j++){
                                if(same == true){
                                    if(ovW == 0 && j < kW - 1){
                                        continue;
                                    }
                                    if(ovW == overlapW - 1 && j > cW - kW){
                                        continue;
                                    }

                                    indA[0] = ovW * k2W + j - (kW - 1);
                                } else {
                                    indA[0] = ovW * k2W + j;
                                }

                                // continue when the input is not a multiple of 2
                                if(indA[0] >= oW)
                                    continue;

                                pInd[0] = j;

                                pI = DoubleTensor.indicesToNum(pInd, dimsP) << 1;
                                ind = DoubleTensor.indicesToNum(indA, dims);
                                output.data.addQuick(ind, outTemp[pI]);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Given an integer, this function obtains the next power of 2. For example
     * nextPowerOf2(3) = 4, nextPowerOf2(4) = 4, nextPowerOf2(5) = 8.
     * @param n Input integer to the function
     * @return Next power of 2
     */
    public static int nextPowerOf2(int n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n++;
        return n;
    }
}