package io;

import tensor.DoubleTensor;

/**
 * Used for scaling data either to standardize (i.e., to have values with
 * zero mean and unit stddev) or to normalize (move values to range [0,1]).
 * @author diego_paez
 */
public class ScaleData {

    /**
     * Store the means of a calculated standardization.
     */
    public DoubleTensor Xmean;
    
    /**
     * Store the standard deviation of a calculated standardization.
     */
    public DoubleTensor Xstd;

    /**
     * Store difference between maximum and minimum of a calculated normalization.
     */
    public DoubleTensor Xdif;
    
    /**
     * Store minimum of a calculated normalization.
     */
    public DoubleTensor Xmin;

    /**
     * Standardize values by subtracting mean and dividing by standard deviation
     * using byDim dimension. Standard deviation is only allowed to have a
     * minimum value of 1E-4 to avoid division by zero. The last i in standardizei
     * means data is standardize in place, i.e., it is stored in given tensor
     * X.
     * @param byDim Dimension by which to standardize.
     * @param X Data to standardize.
     * @return Standardized tensor.
     */
    public DoubleTensor standardizei(int byDim, DoubleTensor X){
 
        Xmean = X.byDimAvg(byDim);
        Xstd = X.byDimStd(byDim);
        Xstd.maxi(1E-4); // to avoid zeros in std

        X.subiLowerDimTensor(byDim, Xmean);
        X.diviLowerDimTensor(byDim, Xstd);

        return X;
    }

    /**
     * Standardize using calculated values in method standardizei.
     * @param byDim Dimension by which to standardize.
     * @param X Data to standardize.
     * @return Standardized tensor.
     */
    public DoubleTensor standardizeiWithCalculated(int byDim, DoubleTensor X){
        X.subiLowerDimTensor(byDim, Xmean);
        X.diviLowerDimTensor(byDim, Xstd);

        return X;
    }

    /**
     * Every value takes a linear transformation from the current range [min,max]
     * to the range [0,1] (this is done byDim dimension). To accomplish this,
     * each value is subtracted by the minimum (of the byDim dimension), and
     * divided by the max minus the min. Values are normalized in place.
     * @param byDim Dimension by which to normalize.
     * @param X Data to standardize.
     * @return Normalized tensor.
     */
    public DoubleTensor normalizei(int byDim, DoubleTensor X){
        Xmin = X.byDimMin(byDim);
        Xdif = X.byDimMax(byDim);
        Xdif.subi(Xmin);
        Xdif.maxi(1E-4); // to avoid zeros in norm

        X.subiLowerDimTensor(byDim, Xmin);
        X.diviLowerDimTensor(byDim, Xdif);

        return X;
    }

    /**
     * Normalize using calculated values in method normalizei.
     * @param byDim Dimension by which to normalize.
     * @param X Data to standardize.
     * @return Normalized tensor.
     */
    public DoubleTensor normalizeiWithCalculated(int byDim, DoubleTensor X){
        X.subiLowerDimTensor(byDim, Xmin);
        X.diviLowerDimTensor(byDim, Xdif);

        return X;
    }
}
