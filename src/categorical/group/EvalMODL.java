package categorical.group;

import java.util.ArrayList;
import org.apache.commons.math3.special.Gamma;

/**
 * Does evalutaion of class grouping based on the grouping method MODL.
 * The algorithm is described in the article:
 * "A Bayes Optimal Approach for Partitioning the Values of Categorical Attributes".
 * by Marc Boullé
 * 
 * @author diego_paez
 */
public class EvalMODL implements EvalGrouping{

    
    /**
     * Evaluates a grouping according to the MODL algorithm. See Theorem 1 in
     * article:
     * log(I) + log(B(I,K)) + sum(k=1,K,log(C(J-1,n_k + J-1))) + ...
     * @param numVals I = number of categorical values
     * @param values Strings as they appear per row.
     * @param numClasses J = number of classes
     * @param classes Strings of each one of the classes
     * @param numGroups Number of groupings
     * @param group Integer values for each grouping
     * @return Measure of how good the grouping is, the lower the better.
     */
    @Override
    public double evalGroup(int numVals, ArrayList<String> values, int numClasses,
            ArrayList<Integer> classes, int numGroups, ArrayList<Integer> group) {
        double total;
        
        // First term in formula
        total = Math.log(numVals);
        
        // Add second term in formula
        total += Math.log(countEl(numVals, numGroups));

        // count classes occuring in each group
        int[] groupsSize = new int[numGroups];
        int[][] classesInGroups = new int[numGroups][numClasses];

        for(int i = 0; i < classes.size(); i++){
            groupsSize[group.get(i)]++;
            classesInGroups[group.get(i)][classes.get(i)]++;
        }

        // Add third and fourth terms
        total += probErr(numClasses, groupsSize, classesInGroups);

        return total;
    }

    /**
     * Computes the third and fourth term of the equation of MODL.
     * @param numClasses Number of Classes
     * @param groupsSize Size of each group
     * @param classesInGroups Size of each class within each groups
     * @return third and fourth term of model
     */
    private double probErr(int numClasses, int[] groupsSize, int[][] classesInGroups){
        int k = numClasses - 1;
        double total = 0;
        for(int i = 0; i < k; i++){

            // log of Binomial coef (third and fourth term are group together to reduce calculations)
            total += computeLogFactorial(groupsSize[i] + k); // num. 3rd term
            // total -= computeLogFactorial(k);             // den. 3rd term (this term has been factored out)
            // total -= computeLogFactorial(groupsSize[i]); // den. 3rd term
            // total += computeLogFactorial(groupsSize[i]); // num. 4th term
            for(int j = 0; j < numClasses; j++){
                total -= computeLogFactorial(classesInGroups[i][j]);
            }
        }
        total -= k * computeLogFactorial(k); // term that was factored out

        return total;
    }

    /**
     * Compute the logarithm of a factorial. Uses log gamma function for numbers
     * above 90, else sums n logarithms of different numbers.
     * @param i The number of which the log factorial will be calcuated.
     * @return 
     */
    private double computeLogFactorial(int i){
        if(i > 90){
            return Gamma.logGamma(i + 1);
        } else{
            double logFact = 0;
            for(int j = 1; j <= i; j++){
                logFact += Math.log(j);
            }
            return logFact;
        }
    }

    /**
     * Count ways n distinguishable elements can be put in c indistinguishable
     * containers.
     * 
     * The algorithm calculates the Stirling numbers of the second kind and
     * then adds them up.
     * https://scicomp.stackexchange.com/questions/1571/an-efficient-way-to-numerically-compute-stirling-numbers-of-the-second-kind
     * 
     * Note that if n == c the bell triangle could be used to calculate the
     * number but since that is usually not the case individual Stirling numbers
     * must be calculated.
     * 
     * @param n Number of distinguishable elements
     * @param c Number of indistinguishable containers
     * @return B(n,c)
     */
    public double countEl(int n, int c){
        double[] prev = new double[c];
        double[] curr = new double[c];

        int k, min;
        prev[0] = 1;
        curr[0] = 1;
        for(int i = 0; i < n; i++){
            min = Math.min(i + 1,c);
            // generate current row of Stirling numbers of the second kind
            k = 1;
            while(k < min){
                curr[k] = prev[k - 1] + (k + 1) * prev[k];
                k++;
            }

            // copy curr into prev
            k = 1;
            while(k < min){
                prev[k] = curr[k];
                k++;
            }
        }

        // Add up the Stirling numbers to get B(n,c)
        double tot = 0;
        for(int i = 0; i < c; i++)
            tot += curr[i];

        return tot;
    }

}
