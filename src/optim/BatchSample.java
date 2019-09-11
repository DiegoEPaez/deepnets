package optim;

import java.util.Arrays;
import java.util.Random;
import tensor.DoubleTensor;
import tensor.TensorIndex;

/**
 *
 * @author diego_paez
 */
public class BatchSample {

    private DoubleTensor X;
    private DoubleTensor y;
    private DoubleTensor yWeights;
    private int batchSize;
    private int batchPointer;
    private int[] rand;


    public BatchSample(DoubleTensor X, DoubleTensor y,
            DoubleTensor yWeights, int batchSize) {
        this.X = X;
        this.y = y;
        this.yWeights = yWeights;
        this.batchSize = batchSize;
    }

    public void initRandBatch(){
        rand = randperm(X.dims[X.dims.length - 1]);
        this.batchPointer = 0;
    }

    public int getNumBatches(){
        double dblCols = (double) X.lastDim();
        double dblSize = (double) batchSize;
        return (int) Math.ceil(dblCols / dblSize);
    }

    public boolean nextBatch(){
        if(batchPointer + batchSize > X.lastDim()){
            return false;
        } else{
            batchPointer += batchSize;
            return true;
        }
    }

    public DoubleTensor getBatchX(){
        int[] currB = Arrays.copyOfRange(rand, batchPointer, Math.min((batchPointer + batchSize),
                X.lastDim()));
        TensorIndex ind = new TensorIndex(currB);
        DoubleTensor XBatch = X.getByDim(X.dims.length - 1,ind);
        return XBatch;
    }

    public DoubleTensor getBatchY(){
        int[] currB = Arrays.copyOfRange(rand, batchPointer, Math.min((batchPointer + batchSize),
                y.lastDim()));
        TensorIndex ind = new TensorIndex(currB);
        DoubleTensor yBatch = y.getByDim(y.dims.length - 1,ind);
        return yBatch;
    }

    public DoubleTensor getBatchT(){
        if(yWeights == null)
            return null;

        int[] currB = Arrays.copyOfRange(rand, batchPointer, Math.min((batchPointer + batchSize),
                yWeights.lastDim()));
        TensorIndex ind = new TensorIndex(currB);
        DoubleTensor tBatch = yWeights.getByDim(yWeights.dims.length - 1,ind);
        return tBatch;
    }

    /**
     * Use Fischer-Yates shuffle "Algorithm P" in David Knuth's (see wikipedia)
     * @param last
     * @return
     */
    public static int[] randperm(int last){
        Random randP = new Random();
        int[] result = new int[last];
        int j;
        for(int i = 0; i < last; i++){
            j = randP.nextInt(i + 1);
            if(j != i){
                result[i] = result[j];
            }
            result[j] = i;
        }

        return result;
    }

}
