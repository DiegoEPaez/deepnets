package eval;

import core.NeuralNetModel;
import java.util.Arrays;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.log4j.Logger;
import tensor.DoubleTensor;
import tensor.TensorIndex;

/**
 * This class has several methods which are useful to evauate the performance
 * of a neural net (or for that matter a machine learning algo.).
 * @author diego_paez
 */
public class EvalUtil {

    /**
     * Logger of log4j.
     */
    private static final Logger LOG = Logger.getLogger(EvalUtil.class); 

    /**
     * Takes a matrix and returns the index of the maximum per column. This 
     * index represents the prediction of a neural net (or a mach. learning algo.).
     * This function assumes each row represents the probability of each possible
     * output and by taking the index which has the maximum per column, the
     * classification of the neural net is returned.
     * @param matrix Matrix which represents the different probabilities a neural
     * net assigns to each example.
     * @return Array with classification of neural net.
     */
    public static double[] classify(double[][] matrix){
        double [] classified = new double[matrix[0].length];
        for(int i = 0; i < matrix[0].length; i++){
            int max = 0;
            for(int j = 0; j < matrix.length;j++){
                if(matrix[j][i] > matrix[max][i]){
                    max = j;
                }
            }
            classified[i] = max;
        }

        return classified;
    }

    /**
     * Takes a vector which has the probability of classifying each example as
     * 0 or 1. If the probability is closer to 1 than to 0 it is classified as
     * 1 and if closer to 0 then it is classified as 0. 
     * @param vector Vector with probabilities of classifying as 1 each example.
     * @return vector with classifications as either 0 or 1.
     */
    public static double[] classify(double[] vector){
        double [] classified = new double[vector.length];
        for(int i = 0; i < classified.length; i++){
            if(vector[i] < 0.5)
                classified[i] = 0.0;
            else
                classified[i] = 1.0;
        }

        return classified;
    }

    /**
     * Takes in a 2D Tensor which has in the first dimension the number of
     * classes, in the second dimension the number of examples and in each entry
     * the probability a neural net assigned to that class for that example.
     * The function outputs the class with the highest probability for each
     * example.
     * 
     * @param predicted 2D Tensor with probabilities for each class.
     * @return Classes predicted for each example.
     */
    public static double[] classify(DoubleTensor predicted){
        if(predicted.dims.length != 2){
            LOG.error("Currently can not classify on a Tensor different than 2D");
            return null;
        }

        double [] classified = new double[predicted.dims[1]];
        int ind1, ind2;
        for(int i = 0; i < predicted.dims[1]; i++){
            int max = 0;
            for(int j = 0; j < predicted.dims[0];j++){
                ind1 = DoubleTensor.indicesToNum(new int[]{j,i}, predicted.dims);
                ind2 = DoubleTensor.indicesToNum(new int[]{max,i}, predicted.dims);
                if(predicted.getQuick(ind1) > predicted.getQuick(ind2)){
                    max = j;
                }
            }
            classified[i] = max;
        }

        return classified;
    }

    /**
     * Count the number of missclasifications performed by a neural net.
     * The function receives 2 parameters a vector containing the predictions
     * and another vector with the actual values of the target variable and counts
     * the number of missclasifications.
     * @param predicted Predicted classes.
     * @param actual Actual classes.
     * @return Number of missclassified.
     */
    public static int missClassified(double[] predicted, double[] actual){
        int total = 0;
        for(int i = 0; i < predicted.length; i++){
            if(predicted[i] != actual[i])
                total++;
        }

        return total;
    }
    
    /**
     * Evaluates the number of missclassified examples from the generated output
     * of a neural net and a set X against a target variable y. The output
     * generate by the neural net is calculated in batches to avoid loading too
     * much information into memory.
     * 
     * @param model Neural net model.
     * @param X Set of independent variables.
     * @param y Target variable.
     * @param batchSize Size of each batch to evaluate
     * @return Amount of examples which were missclassified.
     */
    public static int batchEvalMissClassified(NeuralNetModel model, DoubleTensor X,
            DoubleTensor y, int batchSize){
        DoubleTensor predicted;
        double examples = (double) X.lastDim();
        double dblSize = (double) batchSize;
        int numBatches = (int) Math.ceil(examples / dblSize);
        int batchPointer = 0;
        int numMissClas = 0;
        TensorIndex ind;
        DoubleTensor XBatch, yBatch;

        // for each batch peform an evauation and count number of missclassified.
        for(int i = 0; i < numBatches; i++){
            
            // create a TensorIndex (set of indices) to represent current batch to eval.
            ind = new TensorIndex(batchPointer,Math.min((batchPointer + batchSize),
                    X.lastDim()));
            XBatch = X.getByDim(X.dims.length - 1,ind);
            
            ind.resetPointer();
            yBatch = y.getByDim(y.dims.length - 1,ind);

            predicted = model.fProp(XBatch);
            double[] classified = EvalUtil.classify(predicted);

            double[] actual = yBatch.data.getData();
            numMissClas += EvalUtil.missClassified(classified, actual);

            batchPointer += batchSize;
        }
        return numMissClas;
    }

    /**
     * Calculates the area under ROC curve for the predictions of a neural net
     * in which there are only 2 possible classes. The method takes in the
     * predicted as a 2D array where the first row has the probability of class
     * 0 and the second the probability of class 1, and also takes in an array
     * with actual values.
     * 
     * @param predicted 2D array with probabilties for each class
     * @param actual Actual values
     * @return Area under ROC curve.
     */
    public static double ROC(double[][] predicted, double[] actual){
        return ROC(predicted[1],actual);
    }

    /**
     * Calculates the area under ROC curve for the predictions of a neural net
     * in which there are only 2 possible classes. The first parameter is 
     * the probability of class 1, the second is the actual values (either class
     * 0 or class 1).
     * 
     * @param predicted 2D array with probabilties for each class
     * @param actual Actual values
     * @return Area under ROC curve.
     */
    public static double ROC(double[] predicted, double[] actual){
        double c0 = 0, c1 = 0;
        double a0 = 0, a1 = 0;
        double xPrev = 0, yPrev = 0, x, y, area;

        // Sort data by prices of actual
        Double[] pred = ArrayUtils.toObject(predicted);
        ArrayIndexComparator comparator = new ArrayIndexComparator(pred);
        Integer[] indexes = comparator.createIndexArray();
        Arrays.sort(indexes, comparator);

        // first pass count 0's & 1's
        for(int i = 0; i < predicted.length; i++){
            if(actual[indexes[i]] == 0)
                c0++;
            else if(actual[indexes[i]] == 1)
                c1++;
        }

        // second pass obtain X & Y measure
        area = 0.0;
        for(int i = 0; i < predicted.length; i++){
            if(actual[indexes[i]] == 0)
                a0++;
            else if(actual[indexes[i]] == 1)
                a1++;


            x = a0 / c0;
            y = a1 / c1;

            if(i > 0){
                area += (x - xPrev) * (y + yPrev) / 2.0;
            }
            xPrev = x;
            yPrev = y;
        }

        return area;
    }

    /**
     * Count how many examples were correctly classified in the top 1000.
     * @param predicted
     * @param actual
     * @return 
     */
    public static double top1000(double[] predicted, double[] actual){
        // Sort data by prices of actual
        Double[] pred = ArrayUtils.toObject(predicted);
        ArrayIndexComparator comparator = new ArrayIndexComparator(pred);
        Integer[] indexes = comparator.createIndexArray();
        Arrays.sort(indexes, comparator);

        int total = 0;
        for(int i = 0; i < 1000; i++){
            if(actual[indexes[i]] == 1)
                total++;
        }

        return total;
    }

    /**
     * Count how many examples were correctly classified in the top 1000.
     * @param predicted
     * @param actual
     * @return 
     */
    public static double top1000(double[][] predicted, double[] actual){
        return top1000(predicted[1],actual);
    }

    /**
     * Count how many examples were correctly classified in the top 1000 that
     * have 0 in sarprv array.
     * @param predicted
     * @param actual
     * @param sarprv
     * @return 
     */
    public static double top1000SP(double[] predicted, double[] actual, double[] sarprv){
        // Sort data by prices of actual
        Double[] pred = ArrayUtils.toObject(predicted);
        ArrayIndexComparator comparator = new ArrayIndexComparator(pred);
        Integer[] indexes = comparator.createIndexArray();
        Arrays.sort(indexes, comparator);

        int total = 0;
        for(int i = 0; i < 1000; i++){
            if(actual[indexes[i]] == 1 && sarprv[indexes[i]] == 0)
                total++;
        }

        return total;
    }

    /**
     * Count how many examples were correctly classified in the top 1000 that
     * have 0 in sarprv array.
     * @param predicted
     * @param actual
     * @param sarprv
     * @return 
     */
    public static double top1000SP(double[][] predicted, double[] actual,double[] sarprv){
        return top1000SP(predicted[1],actual,sarprv);
    }
}
