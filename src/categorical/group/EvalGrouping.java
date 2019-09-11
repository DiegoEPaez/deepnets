package categorical.group;

import java.util.ArrayList;

/**
 * This interface specifies the necessary methods to evaluate how good is a 
 * grouping of categorical clases of a categorical variable. The need to group
 * classes of a categorical variables arises from the fact that when a categorical
 * variable has too many classes the predictive power of this variable is small
 * but by grouping the many classes a higher predictive power may be achieved.
 * @author diego_paez
 */
public interface EvalGrouping {
    
    /**
     * Evaluates a grouping of classes of a categorical variable.
     * @param numVals Number of categorical values
     * @param values Strings as they appear per row.
     * @param numClasses Number of classes
     * @param classes Strings of each one of the classes
     * @param numGroups nNumber of groupings
     * @param group Integer values for each grouping
     * @return Measure of how good the grouping is, the lower the better.
     */
    public double evalGroup(int numVals, ArrayList<String> values,
            int numClasses, ArrayList<Integer> classes,
            int numGroups, ArrayList<Integer> group);
}
