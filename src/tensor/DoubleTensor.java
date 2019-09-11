package tensor;

import org.apache.log4j.Logger;
import org.jblas.ComplexDouble;
import org.jblas.NativeBlas;

/**
 * Represent an n-dimensional Tensor numbers appear first advancing the first
 * dimension, then the second and so on!
 * @author diego_paez
 */
public class DoubleTensor {

    public DoubleArrayList data;
    public int[] dims;

    private static final Logger LOG = Logger.getLogger(DoubleTensor.class);
    public static final ComplexDouble CZERO = new ComplexDouble(0.0);
    public static final ComplexDouble CONE = new ComplexDouble(1.0);

    public enum MathOperations {
        ADD,
        SUB,
        RSUB,
        MUL,
        MAX,
        MIN,
        DIV,
        RDIV;
    }

    public enum AggregateOper{
        SUM,
        SUMSQ,
        MIN,
        MAX,
        PRODUCT
    }

    public DoubleTensor(int... dims){
        this.dims = new int[dims.length];

        int length = 1;
        for (int i = 0; i < dims.length; ++i){
            this.dims[i] = dims[i];
            length *= dims[i];
        }
        data = new DoubleArrayList(length);
    }

    /**
     * Stores data as is
     * @param data
     * @param dims
     */
    public DoubleTensor(double[] data, int...dims){
        this.dims = dims;
        this.data = new DoubleArrayList(data);
    }


    /**
     * This constructor copies data from other ArrayList!
     * @param initData
     * @param dims
     */
    public DoubleTensor(DoubleArrayList initData, int... dims){
        
        this.dims = new int[dims.length];
        int length = 1;
        for (int i = 0; i < dims.length; ++i){
            this.dims[i] = dims[i];
            length *= dims[i];
        }

        if(length != initData.size()){
            LOG.error("Tensor constructor requires initData and dims to match in size, but they don't: "
                    + "initData size: " + initData.size() + ", dims size: " + length + " creating"
                    + " a one dimensional tensor with size = initData");
            dims = new int[1];
            dims[0] = length;
        }
        data = new DoubleArrayList(initData);

    }

    public int size() {
        return data.size();
    }

    public int length(){
        int length = 1;
        for (int i = 0; i < dims.length; i++){
            length *= dims[i];
        }

        return length;
    }

    public void setPosToLast(){
        data.setPos(data.getData().length);
    }

    public void setPos(int pos){
        data.setPos(pos);
    }


    public void reshape(int... dims){
        this.dims = new int[dims.length];

        int length = 1;
        for (int i = 0; i < dims.length; i++){
            this.dims[i] = dims[i];
            length *= dims[i];
        }

        if (length != size()){
            LOG.error("Dimension must match size");
        }
    }

    public void resize(int... dims){
        this.dims = new int[dims.length];

        int length = 1;
        for (int i = 0; i < dims.length; i++)
        {
            this.dims[i] = dims[i];
            length *= dims[i];
        }
        data.ensureCapacity(length);

    }
    
    public static int[] shapeDims2d(int[] dims){
        int[] dims2d = new int[2];
        dims2d[0] = 1;
        for(int i = 0; i < dims.length; i++){
            if(i < dims.length - 1){
                dims2d[0] *= dims[i];
            } else{
                dims2d[1] = dims[i];
            }
        }

        return dims2d;
    }

    private boolean dimensionCheck(DoubleTensor t1, DoubleTensor t2){
        if(t1.dims.length != t2.dims.length){
            LOG.warn("Tensor 1 has " + t1.dims.length + " dimensions " +
                    "Tensor 2 has " + t2.dims.length);
            return false;
        }
        for(int i = 0; i < t1.dims.length; i++){
            if(t1.dims[i] != t2.dims[i]){
                LOG.warn("Tensor 1 has dimension " + i + " size: " + t1.dims[i] +
                    " Tensor 2 has dimension " + i + " size: " + t2.dims[i]);
            return false;
            }
        }

        return true;
    }

    public int[] copyDims(){
        int[] copy = new int[this.dims.length];
        System.arraycopy(dims, 0, copy, 0, dims.length);

        return copy;
    }

    /**
     * Makes a copy of this tensor
     * @return
     */
    public DoubleTensor copy(){
        int[] cDims = copyDims();
        
        DoubleArrayList cList = new DoubleArrayList(this.size());

        for(int i = 0; i < this.size(); i++){
            cList.add(this.data.getQuick(i));
        }

        DoubleTensor cTensor = new DoubleTensor(cList,cDims);

        return cTensor;
    }

    private static int product(int[] dims){
        int total = 1;
        for(int i = 0; i < dims.length; i++){
            total *= dims[i];
        }
        return total;
    }

    public void setQuick(int index, double d){
        data.setQuick(index, d);
    }

    public double getQuick(int index){
        return data.getQuick(index);
    }

    /**
     * Data in the tensor is stored as a 1D array. This method transforms
     * the position in the 1D array to a position in the n-dimensions.
     *
     * For 3 dims given a number n:
     * i = n % d1
     * j = (i / d1) % d2
     * k = i / d1 / d2
     * @param num
     * @param dims
     * @return
     */
    public static void numToIndices(int num, int[] dims, int[] indices){
        // sanity check
        if(dims == null){
            LOG.error("Dims is null");
            return;
        }
        if(num < 0 || num >= product(dims)){
            LOG.error("num out of bounds");
            return;
        }
        if(indices == null || indices.length < dims.length){
            LOG.error("indices == null or too small");
            return;
        }
        // end sanity check

        int accum = num;
        for(int i = 0; i < dims.length; i++){
            indices[i] = accum % dims[i];
            accum = accum / dims[i];
        }
    }

    /**
     * Data in the tensor is stored as a 1D array. This method transforms
     * the position in an n-D array to a position in 1D.
     *
     * For dimensions, given i,j,k:
     * 1d-pos = i + d1 * j + d1 * d2 * k =
     * 1
     * * (i + d1
     * * (j + d2
     * * k
     *
     * @param indices
     * @param dims
     * @return
     */
    public static int indicesToNum(int indices[], int[] dims){
        // sanity check
        if(indices == null || dims == null || indices.length != dims.length){
            LOG.error("Can't convert: either indices or dims are null or their lengths do not match");
            return -1;
        }

        for(int i = 0; i < dims.length; i++){
           if(indices[i] < 0 || indices[i] >= dims[i]){
               LOG.error("Can't convert: either some index is less than zero or is bigger than a dimension");
               return -1;
            }
        }
        // end sanity check

        int num = 0;
        int accum;
        accum = 1;
        for(int i = 0; i < dims.length; i++){
            num += accum * indices[i];
            accum *= dims[i];
        }

        return num;
    }

    /**
     * Add a scalar
     * @param scalar
     */
    public DoubleTensor operi(double scalar, MathOperations oper, DoubleTensor store){

        if(store == null){
            store = new DoubleTensor(this.copyDims());
            store.data.setPos(store.data.getData().length);
        }

        for(int i = 0; i < this.size(); i++){
            if(oper == MathOperations.ADD)
                store.data.setQuick(i, data.getQuick(i) + scalar);
            else if(oper == MathOperations.SUB)
                store.data.setQuick(i, data.getQuick(i) - scalar);
            else if(oper == MathOperations.RSUB)
                store.data.setQuick(i, scalar - data.getQuick(i));
            else if (oper == MathOperations.MUL)
                store.data.setQuick(i, data.getQuick(i) * scalar);
            else if (oper == MathOperations.DIV)
                store.data.setQuick(i, data.getQuick(i) / scalar);
            else if (oper == MathOperations.RDIV)
                store.data.setQuick(i, scalar / data.getQuick(i));
            else if (oper == MathOperations.MAX)
                store.data.setQuick(i, Math.max(scalar, data.getQuick(i)));
            else // MIN
                store.data.setQuick(i, Math.min(scalar, data.getQuick(i)));
        }
        return store;
    }

   
    /**
     * Add a Tensor
     * @param x
     */
    private DoubleTensor operi(DoubleTensor x, MathOperations oper, DoubleTensor store){
        // perform dimension check
        if(!dimensionCheck(this, x)){
            LOG.error("Can't operate, dimensions do not match");
        }

        if(store == null){
            store = new DoubleTensor(this.copyDims());
            store.data.setPos(store.data.getData().length);
        }

        for(int i = 0; i < this.size(); i++){
            if(oper == MathOperations.ADD)
                store.data.setQuick(i, data.getQuick(i) + x.data.getQuick(i));
            else if(oper == MathOperations.SUB)
                store.data.setQuick(i, data.getQuick(i) - x.data.getQuick(i));
            else if(oper == MathOperations.RSUB)
                store.data.setQuick(i, x.data.getQuick(i) - data.getQuick(i));
            else if (oper == MathOperations.MUL)
                store.data.setQuick(i, data.getQuick(i) * x.data.getQuick(i));
            else if (oper == MathOperations.DIV)
                store.data.setQuick(i, data.getQuick(i) / x.data.getQuick(i));
            else if (oper == MathOperations.RDIV)
                store.data.setQuick(i, x.data.getQuick(i) / data.getQuick(i));
            else if (oper == MathOperations.MAX)
                store.data.setQuick(i, Math.max(data.getQuick(i),x.data.getQuick(i)));
            else // MIN
                store.data.setQuick(i, Math.min(data.getQuick(i),x.data.getQuick(i)));
        }
        return store;
    }

    /**
     * returns a lower dimensional Tensor (by 1 dim. lower) which is the
     * aggregation of this Tensor by the specified dimension: aggDim.
     * @param aggDim
     * @param oper
     * @param store
     * @return
     */
    private DoubleTensor dimAggregatei(int aggDim, AggregateOper oper, DoubleTensor store){
        // sanity check
        if(aggDim < 0 || aggDim >= dims.length){
            LOG.error("Can't aggregate on given dimension");
        }
        // end sanity check

        int[] dimsAg = new int[dims.length - 1];
        int lenAg = 1;
        for(int i = 0, j = 0; i < dims.length; i++){
            if(i == aggDim)
                continue;
            dimsAg[j] = dims[i];
            lenAg *= dims[i];
            j++;
        }

        DoubleTensor aggregated;
        if(store == null){
            aggregated = new DoubleTensor(dimsAg);
            aggregated.setPosToLast();
        } else{
            aggregated = store;
        }

        // init values
        for(int i = 0; i < lenAg; i++){
            if(oper == AggregateOper.SUM || oper == AggregateOper.SUMSQ){
                aggregated.data.setQuick(i,0.0);
            } else if (oper == AggregateOper.PRODUCT){
                aggregated.data.setQuick(i,1.0);
            } else if (oper == AggregateOper.MIN){
                aggregated.data.setQuick(i,Double.MAX_VALUE);
            } else {
                aggregated.data.setQuick(i,-Double.MAX_VALUE);
            }
        }

        int[] indAgg = new int[dimsAg.length];
        int[] indTen = new int[dims.length];
        int iAgg, iTen;

        for(int i = 0; i < dims[aggDim]; i++){
            for(int j = 0; j < aggregated.size(); j++){
                numToIndices(j, dimsAg, indTen);

                System.arraycopy(indTen, 0, indAgg, 0, dimsAg.length);
                // insert i
                for(int k = dims.length - 1; k > aggDim; k--){
                    indTen[k] = indTen[k - 1];
                }
                indTen[aggDim] = i;

                iAgg = indicesToNum(indAgg, dimsAg);
                iTen = indicesToNum(indTen, dims);

                if(oper == AggregateOper.SUM){
                    aggregated.data.setQuick(iAgg, aggregated.data.getQuick(iAgg) +
                            data.getQuick(iTen));
                } else if(oper == AggregateOper.SUMSQ){
                    aggregated.data.setQuick(iAgg, aggregated.data.getQuick(iAgg) +
                            data.getQuick(iTen) * data.getQuick(iTen));
                } else if(oper == AggregateOper.PRODUCT){
                    aggregated.data.setQuick(iAgg, aggregated.data.getQuick(iAgg) *
                            data.getQuick(iTen));
                } else if(oper == AggregateOper.MAX){
                    aggregated.data.setQuick(iAgg, Math.max(aggregated.data.getQuick(iAgg),
                            data.getQuick(iTen)));
                } else if(oper == AggregateOper.MIN){ // MIN
                    aggregated.data.setQuick(iAgg, Math.min(aggregated.data.getQuick(iAgg),
                            data.getQuick(iTen)));
                }
            }
        }

        return aggregated;
    }

    /**
     * Aggregates by all dimensions except one.
     * @param aggDim
     * @param oper
     * @param store
     * @return
     */
    private DoubleTensor aggregateAlli(int aggDim, AggregateOper oper, DoubleTensor store){
        // sanity check
        if(aggDim < 0 || aggDim >= dims.length){
            LOG.error("Can't aggregate on given dimension");
        }
        // end sanity check

        int[] dimsRem = new int[dims.length - 1];
        int lenRem = 1;
        for(int i = 0, j = 0; i < dims.length; i++){
            if(i == aggDim)
                continue;
            dimsRem[j] = dims[i];
            lenRem *= dims[i];
            j++;
        }

        DoubleTensor aggregated;
        if(store == null){
            aggregated = new DoubleTensor(dims[aggDim]);
            aggregated.setPosToLast();
        } else{
            aggregated = store;

            for(int i = 0; i < dims[aggDim]; i++){
                if(oper == AggregateOper.SUM || oper == AggregateOper.SUMSQ){
                    aggregated.data.setQuick(i,0.0);
                } else if (oper == AggregateOper.PRODUCT){
                    aggregated.data.setQuick(i,1.0);
                } else if (oper == AggregateOper.MIN){
                    aggregated.data.setQuick(i,Double.MAX_VALUE);
                } else {
                    aggregated.data.setQuick(i,-Double.MAX_VALUE);
                }
            }

        }

        int[] indTen = new int[dims.length];
        int iTen;

        for(int iAgg = 0; iAgg < dims[aggDim]; iAgg++){
            for(int j = 0; j < lenRem; j++){
                numToIndices(j, dimsRem, indTen);

                // insert i
                for(int k = dims.length - 1; k > aggDim; k--){
                    indTen[k] = indTen[k - 1];
                }
                indTen[aggDim] = iAgg;

                iTen = indicesToNum(indTen, dims);

                if(oper == AggregateOper.SUM){
                    aggregated.data.setQuick(iAgg, aggregated.data.getQuick(iAgg) +
                            data.getQuick(iTen));
                } else if(oper == AggregateOper.SUMSQ){
                    aggregated.data.setQuick(iAgg, aggregated.data.getQuick(iAgg) +
                            data.getQuick(iTen) * data.getQuick(iTen));
                } else if(oper == AggregateOper.PRODUCT){
                    aggregated.data.setQuick(iAgg, aggregated.data.getQuick(iAgg) *
                            data.getQuick(iTen));
                } else if(oper == AggregateOper.MAX){
                    aggregated.data.setQuick(iAgg, Math.max(aggregated.data.getQuick(iAgg),
                            data.getQuick(iTen)));
                } else if(oper == AggregateOper.MIN){ // MIN
                    aggregated.data.setQuick(iAgg, Math.min(aggregated.data.getQuick(iAgg),
                            data.getQuick(iTen)));
                }
            }
        }

        return aggregated;
    }

    /**
     * Obtains a lower dimension tensor from a given tensor using an array of
     * indices for a fixed dimension. For example given the 2D Tensor:
     *
     * 1 2 3
     * 4 5 6
     * 7 8 9
     *
     * And the 1D array of indices:
     * 0 2 0
     *
     * Then if the fixedDim = 0, it should return:
     * 1 6 7
     *
     * @param fixedDim
     * @param indices
     * @return
     */
    public DoubleTensor getLowerDim(int aggDim, DoubleTensor dimIndices){
        // sanity check
        if(aggDim < 0 || aggDim >= dims.length){
            LOG.error("Can't aggregate on given dimension");
        }
        // end sanity check

        int[] dimsAg = new int[dims.length - 1];
        int lenAg = 1;
        for(int i = 0, j = 0; i < dims.length; i++){
            if(i == aggDim)
                continue;
            dimsAg[j] = dims[i];
            lenAg *= dims[i];
            j++;
        }

        DoubleTensor res = new DoubleTensor(dimsAg);
        res.data.setPos(lenAg);
        int[] indAgg = new int[dimsAg.length];
        int[] indTen = new int[dims.length];
        int iTen;

        for(int j = 0; j < lenAg; j++){
            numToIndices(j, dimsAg, indTen);

            System.arraycopy(indTen, 0, indAgg, 0, dimsAg.length);
            // insert dimIndices[j]
            for(int k = dims.length - 1; k > aggDim; k--){
                indTen[k] = indTen[k - 1];
            }
            indTen[aggDim] = (int)dimIndices.data.getQuick(j);

            iTen = indicesToNum(indTen, dims);

            res.data.setQuick(j, data.getQuick(iTen));
        }

        return res;
    }

    public int lastDim(){
        return dims[dims.length - 1];
    }

    /**
     * Constructor an n dimensional Tensor from an n dimensional Tensor by getting
     * all n - 1 dimensional Tensor indicated by indices.
     * @param dim
     * @param indices
     */
    public DoubleTensor getByDim(int dim, TensorIndex index){
        // sanity check
        if(dim < 0 || dim >= dims.length){
            LOG.error("Can't aggregate on given dimension");
        }
        // end sanity check

        // dimensions sizes except for the fixed dim
        int[] dimsLow = new int[dims.length - 1];

        // all dimensions, in the fixed dim subs indices.length
        int[] dimsDest = new int[dims.length];
        int lenAg = 1;
        for(int i = 0, j = 0; i < dims.length; i++){
            if(i == dim){
                dimsDest[i] = index.size();
                continue;
            }
            dimsDest[i] = dims[i];
            dimsLow[j] = dims[i];
            lenAg *= dims[i];
            j++;
        }

        DoubleTensor res = new DoubleTensor(dimsDest);
        res.data.setPos(lenAg * index.size());
        int[] indOrig = new int[dims.length];
        int[] indDest = new int[dims.length];
        int iOrig, iDest, iDim;

        for(int i = 0; i < index.size(); i++){
            iDim = index.next();
            for(int j = 0; j < lenAg; j++){
                numToIndices(j, dimsLow, indOrig);

                // insert i
                for(int k = dims.length - 1; k > dim; k--){
                    indOrig[k] = indOrig[k - 1];
                }
                indOrig[dim] = iDim;
                System.arraycopy(indOrig, 0, indDest, 0, dims.length);
                indDest[dim] = i;

                iOrig = indicesToNum(indOrig, dims);
                iDest = indicesToNum(indDest, dimsDest);

                res.data.setQuick(iDest, data.getQuick(iOrig));
            }
        }

        return res;
    }

    /**
     * Creates an indicator for each entry of this Tensor, i.e., given an
     * n dimensional Tensor constructs an (n + 1)th dim Tensor which is an
     * indicator Tensor.
     * @param numClasses
     * @return
     */
    public DoubleTensor index(int numClasses){
        int[] dimsExp = new int[dims.length + 1];
        int lenPl = numClasses;
        dimsExp[0] = numClasses;
        for(int i = 0, j = 1; i < dims.length; i++, j++){
            lenPl *= dims[i];
            dimsExp[j] = dims[i];
        }

        DoubleTensor res = new DoubleTensor(dimsExp);
        res.data.setPos(lenPl);
        int[] indExp = new int[dimsExp.length];
        int[] indTen = new int[dims.length];
        int iExp;

        for(int j = 0; j < data.size(); j++){
            numToIndices(j, dims, indTen);

            System.arraycopy(indTen, 0, indExp, 1, dims.length);
            // insert in first dim the indicator
            indExp[0] = (int)data.getQuick(j);
            iExp = indicesToNum(indExp, dimsExp);

            res.data.setQuick(iExp, 1.0);
        }

        return res;
    }

    /**
     * Sum all entries
     * @return
     */
    public double sum(){
        double total = 0;
        for(int i = 0; i < data.size(); i++){
            total += data.getQuick(i);
        }

        return total;
    }

    private static DoubleTensor constTensor(double c, int... dims){
        DoubleTensor result = new DoubleTensor(dims);

        for(int i = 0; i < result.length();i++){
            result.data.add(c);
        }

        return result;
    }

    /**
     * Use operation from a Tensor with a dimension less than this Tensor to all
     * other dimensions.
     * Generalization of adding a vector to all columns or all rows of a matrix.
     * 
     * @param fixedDim dimension which remains fixed
     * @param x
     * @return
     */
    private DoubleTensor operiLowerDimTensor(int fixedDim, DoubleTensor x, DoubleTensor store,
            MathOperations oper){
        // sanity check
        for(int i = 0, j = 0; i < dims.length;i++){
            if(i == fixedDim){
                continue;
            }
            if(dims[i] != x.dims[j]){
                LOG.error("Can't operate, dimensions do not match");
                return null;
            }
            j++;
        }
        // end sanity check

        if(store == null){
            store = new DoubleTensor(this.copyDims());
            store.data.setPos(store.data.getData().length);
        }

        int[] indices = new int[dims.length];
        int dataNum;

        for(int i = 0; i < dims[fixedDim]; i++){
            for(int j = 0; j < x.size(); j++){
                numToIndices(j, x.dims, indices);
                
                // insert i
                for(int k = dims.length - 1; k > fixedDim; k--){
                    indices[k] = indices[k - 1];
                }
                indices[fixedDim] = i;

                dataNum = indicesToNum(indices, dims);

                if(oper == MathOperations.ADD)
                    store.data.setQuick(dataNum, data.getQuick(dataNum) + x.data.getQuick(j));
                else if(oper == MathOperations.SUB)
                    store.data.setQuick(dataNum, data.getQuick(dataNum) - x.data.getQuick(j));
                else if(oper == MathOperations.RSUB)
                    store.data.setQuick(dataNum, x.data.getQuick(j) - data.getQuick(dataNum));
                else if (oper == MathOperations.MUL)
                    store.data.setQuick(dataNum, data.getQuick(dataNum) * x.data.getQuick(j));
                else if (oper == MathOperations.DIV)
                    store.data.setQuick(dataNum, data.getQuick(dataNum) / x.data.getQuick(j));
                else
                    store.data.setQuick(dataNum, x.data.getQuick(j) / data.getQuick(dataNum));
            }
        }
        return store;
    }

    /**
     * Perform 2d matrix multiplication
     * @param trans
     * @param transOther
     * @param other
     * @param store
     * @return
     */
    public DoubleTensor mmuli(boolean trans, boolean transOther, DoubleTensor other,
            DoubleTensor store){
        // sanity check
        if(this.dims.length != other.dims.length){
            LOG.error("Dimension mismatch");
            return null;
        }

        if(this.dims.length > 2){
            LOG.error("Matrix multiplication can only be performed on matrices or vectors");
            return null;
        }
        // end checks

        double[] aMat = this.data.getData();
        double[] bMat = other.data.getData();
        double[] cMat;

        // As specified in LAPACK documentation for dgemm, m = rows of C, n = cols of C, k = mult dim
        int  m, n, k;
        char aT, bT;

        if(trans){
            m = this.dims[1]; // columns
            k = this.dims[0]; // rows
            aT = 'T';
        } else{
            m = this.dims[0]; // rows
            k = this.dims[1]; // columns
            aT = 'N';
        }

        if(transOther){
            if(k != other.dims[1]){
                LOG.error("Mismatch in mult dimensions");
                return null;
            }
            n = other.dims[0]; // rows of other
            bT = 'T';
        } else{
            if(k != other.dims[0]){
                LOG.error("Mismatch in mult dimensions");
                return null;
            }
            n = other.dims[1]; // columns of other
            bT = 'N';
        }

        if(store == null){
            cMat = new double[m * n];

            NativeBlas.dgemm(aT, bT, m, n, k, 1.0, aMat, 0, this.dims[0], bMat, 0,
                other.dims[0], 0.0, cMat, 0, m);

            return new DoubleTensor(cMat, new int[]{m,n});
        } else{
            cMat = store.data.getData();
            if(cMat.length < m * n){ // check if cMat has at least the required space
                LOG.error("Output matrix does not have correct size: "+ m + " " + n + "");
                return null;
            }

            NativeBlas.dgemm(aT, bT, m, n, k, 1.0, aMat, 0, this.dims[0], bMat, 0,
                other.dims[0], 0.0, cMat, 0, m);

            return store;
        }
    }

    /**
     * Perform 2d complex matrix multiplication
     * @param trans
     * @param transOther
     * @param other
     * @param store
     * @return
     */
    public DoubleTensor mmulci(boolean trans, boolean transOther, DoubleTensor other,
            DoubleTensor store){
        // sanity check
        if(this.dims.length != other.dims.length){
            LOG.error("Dimension mismatch");
            return null;
        }

        if(this.dims.length > 3 && this.dims[2] != 2){ // last dim = 2
            LOG.error("Matrix multiplication can only be performed on matrices or vectors");
            return null;
        }
        // end checks

        double[] aMat = this.data.getData();
        double[] bMat = other.data.getData();
        double[] cMat;

        // As specified in LAPACK documentation for zgemm, m = rows of C, n = cols of C, k = mult dim
        int  m, n, k;
        char aT, bT;

        if(trans){
            m = this.dims[1]; // columns
            k = this.dims[0]; // rows
            aT = 'T';
        } else{
            m = this.dims[0]; // rows
            k = this.dims[1]; // columns
            aT = 'N';
        }

        if(transOther){
            if(k != other.dims[1]){
                LOG.error("Mismatch in mult dimensions");
                return null;
            }
            n = other.dims[0]; // rows of other
            bT = 'T';
        } else{
            if(k != other.dims[0]){
                LOG.error("Mismatch in mult dimensions");
                return null;
            }
            n = other.dims[1]; // columns of other
            bT = 'N';
        }

        if(store == null){
            cMat = new double[m * n];

            NativeBlas.zgemm(aT, bT, m, n, k, CONE, aMat, 0, this.dims[0], bMat, 0,
                other.dims[0], CZERO, cMat, 0, m);

            return new DoubleTensor(cMat, new int[]{m,n});
        } else{
            cMat = store.data.getData();
            if(cMat.length < m * n * 2){ // check if cMat has at least the required space
                LOG.error("Output matrix does not have correct size: "+ m + " " + n + "");
                return null;
            }

            NativeBlas.zgemm(aT, bT, m, n, k, CONE, aMat, 0, this.dims[0], bMat, 0,
                other.dims[0], CZERO, cMat, 0, m);

            return store;
        }
    }

    public DoubleTensor add(DoubleTensor x){
        return operi(x,MathOperations.ADD,null);
    }

    public DoubleTensor sub(DoubleTensor x){
        return operi(x,MathOperations.SUB,null);
    }

    public DoubleTensor rsub(DoubleTensor x){
        return operi(x,MathOperations.RSUB,null);
    }

    public DoubleTensor mul(DoubleTensor x){
        return operi(x,MathOperations.MUL,null);
    }

    public DoubleTensor div(DoubleTensor x){
        return operi(x,MathOperations.DIV,null);
    }

    public DoubleTensor rdiv(DoubleTensor x){
        return operi(x,MathOperations.RDIV,null);
    }

    public DoubleTensor min(DoubleTensor x){
        return operi(x,MathOperations.MIN,null);
    }

    public DoubleTensor max(DoubleTensor x){
        return operi(x,MathOperations.MAX,null);
    }

    public DoubleTensor addi(DoubleTensor x){
        return operi(x,MathOperations.ADD,this);
    }

    public DoubleTensor subi(DoubleTensor x){
        return operi(x,MathOperations.SUB,this);
    }

    public DoubleTensor rsubi(DoubleTensor x){
        return operi(x,MathOperations.RSUB,this);
    }

    public DoubleTensor muli(DoubleTensor x){
        return operi(x,MathOperations.MUL,this);
    }

    public DoubleTensor divi(DoubleTensor x){
        return operi(x,MathOperations.DIV,this);
    }

    public DoubleTensor rdivi(DoubleTensor x){
        return operi(x,MathOperations.RDIV,this);
    }

    public DoubleTensor mini(DoubleTensor x){
        return operi(x,MathOperations.MIN,this);
    }

    public DoubleTensor maxi(DoubleTensor x){
        return operi(x,MathOperations.MAX,this);
    }

    public DoubleTensor addi(double x){
        return operi(x,MathOperations.ADD,this);
    }

    public DoubleTensor subi(double x){
        return operi(x,MathOperations.SUB,this);
    }

    public DoubleTensor rsubi(double x){
        return operi(x,MathOperations.RSUB,this);
    }

    public DoubleTensor muli(double x){
        return operi(x,MathOperations.MUL,this);
    }

    public DoubleTensor divi(double x){
        return operi(x,MathOperations.DIV,this);
    }

    public DoubleTensor rdivi(double x){
        return operi(x,MathOperations.RDIV,this);
    }

    public DoubleTensor mini(double x){
        return operi(x,MathOperations.MIN,this);
    }

    public DoubleTensor maxi(double x){
        return operi(x,MathOperations.MAX,this);
    }

    public DoubleTensor add(double x){
        return operi(x,MathOperations.ADD,null);
    }

    public DoubleTensor sub(double x){
        return operi(x,MathOperations.SUB,null);
    }

    public DoubleTensor rsub(double x){
        return operi(x,MathOperations.RSUB,null);
    }

    public DoubleTensor mul(double x){
        return operi(x,MathOperations.MUL,null);
    }

    public DoubleTensor div(double x){
        return operi(x,MathOperations.DIV,null);
    }

    public DoubleTensor rdiv(double x){
        return operi(x,MathOperations.RDIV,null);
    }

    public DoubleTensor min(double x){
        return operi(x,MathOperations.MIN,null);
    }

    public DoubleTensor max(double x){
        return operi(x,MathOperations.MAX,null);
    }

    public DoubleTensor addiLowerDimTensor(int fixedDim, DoubleTensor x){
        return operiLowerDimTensor(fixedDim, x, this, MathOperations.ADD);
    }

    public DoubleTensor subiLowerDimTensor(int fixedDim, DoubleTensor x){
        return operiLowerDimTensor(fixedDim, x, this, MathOperations.SUB);
    }

    public DoubleTensor subiLowerDimTensor(DoubleTensor inPlace, int fixedDim, DoubleTensor x){
        return operiLowerDimTensor(fixedDim, x, inPlace, MathOperations.SUB);
    }

    public DoubleTensor rsubiLowerDimTensor(int fixedDim, DoubleTensor x){
        return operiLowerDimTensor(fixedDim, x, this, MathOperations.RSUB);
    }

    public DoubleTensor muliLowerDimTensor(int fixedDim, DoubleTensor x){
        return operiLowerDimTensor(fixedDim, x, this, MathOperations.MUL);
    }

    public DoubleTensor diviLowerDimTensor(int fixedDim, DoubleTensor x){
        return operiLowerDimTensor(fixedDim, x, this, MathOperations.DIV);
    }

    public DoubleTensor rdiviLowerDimTensor(int fixedDim, DoubleTensor x){
        return operiLowerDimTensor(fixedDim, x, this, MathOperations.RDIV);
    }

    public DoubleTensor byDimSum(int aggDim){
        return dimAggregatei(aggDim, DoubleTensor.AggregateOper.SUM, null);
    }

    public DoubleTensor byDimProduct(int aggDim){
        return dimAggregatei(aggDim, DoubleTensor.AggregateOper.PRODUCT, null);
    }

    public DoubleTensor byDimMin(int aggDim){
        return dimAggregatei(aggDim, DoubleTensor.AggregateOper.MIN, null);
    }

    public DoubleTensor byDimMax(int aggDim){
        return dimAggregatei(aggDim, DoubleTensor.AggregateOper.MAX, null);
    }

    public DoubleTensor byDimSumi(int aggDim, DoubleTensor store){
        return dimAggregatei(aggDim, DoubleTensor.AggregateOper.SUM,store);
    }

    public DoubleTensor byDimProducti(int aggDim, DoubleTensor store){
        return dimAggregatei(aggDim, DoubleTensor.AggregateOper.PRODUCT, store);
    }

    public DoubleTensor byDimMini(int aggDim, DoubleTensor store){
        return dimAggregatei(aggDim, DoubleTensor.AggregateOper.MIN, store);
    }

    public DoubleTensor byDimMaxi(int aggDim, DoubleTensor store){
        return dimAggregatei(aggDim, DoubleTensor.AggregateOper.MAX, store);
    }

    public DoubleTensor byDimAvgi(int aggDim, DoubleTensor store){
        DoubleTensor res = dimAggregatei(aggDim, DoubleTensor.AggregateOper.SUM, store);
        return res.divi(this.dims[aggDim]);
    }

    public DoubleTensor byDimAvg(int aggDim){
        return byDimAvgi(aggDim, null);
    }

    /**
     * std^2
     * = 1 / (n-1) * sum(i=0, n, (x_i - x_bar)^2)
     * = 1 / (n-1) * sum(i=0, n, x_i^2 - 2 * x_i * x_bar + x_bar^2)
     * = 1 / (n-1) * [n * E[X^2] - 2 * n * x_bar^2 + n * x_bar^2)]
     * = 1 / (n-1) * (n * E[X^2] - n * x_bar^2)
     * = n / (n-1) * (E[X^2] - x_bar^2)
     * @param aggDim
     * @param store
     * @return
     */
    public DoubleTensor byDimStdi(int aggDim, DoubleTensor store){
        DoubleTensor res = dimAggregatei(aggDim, DoubleTensor.AggregateOper.SUMSQ, store);
        DoubleTensor res2 = dimAggregatei(aggDim, DoubleTensor.AggregateOper.SUM, null);
        res2.muli(res2).divi(this.dims[aggDim]);
        res.subi(res2);
        res.divi((this.dims[aggDim] - 1));

        TensorFunctions.sqrti(res);
        return res;
    }

    public DoubleTensor byDimStd(int aggDim){
        return byDimStdi(aggDim, null);
    }

    public DoubleTensor sumAll(int aggDim){
        return aggregateAlli(aggDim, AggregateOper.SUM, null);
    }

    public DoubleTensor sumAlli(int aggDim, DoubleTensor store){
        return aggregateAlli(aggDim, AggregateOper.SUM, store);
    }

    public DoubleTensor mmul(boolean trans, boolean transOther, DoubleTensor other){
        return mmuli(trans, transOther, other, null);
    }

    public DoubleTensor mmul(DoubleTensor other){
        return mmul(false,false,other);
    }

    public static DoubleTensor ones(int... dims){
        return constTensor(1.0, dims);
    }

    public static DoubleTensor zeros(int... dims){
        return constTensor(0.0, dims);
    }
}
