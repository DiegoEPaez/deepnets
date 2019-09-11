/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package categorical.trans;

import types.DataTypeProperties;
import java.util.ArrayList;

/**
 * Interface for specifying a Categorical to Numeric transformation rule.
 * Defines two methods:
 * replace: given a string return the numeric value (as a String) to assign to each
 * of the columns that will replace the value. So for example if the rule is to
 * use a indicator dummy variable then given the String "A" it might be replaced by
 * {"1","0","0"}
 *
 * generateHeaders: return the headers of the new file by errasing the categorical
 * columns' headers and inserting the headers of the columns that will replace
 * these variables
 * @author diego_paez
 */
public interface TransformCategoricalRule {
    public String[] replace(String str);
    public ArrayList<String> generateHeaders(ArrayList<String> colNames,
            ArrayList<DataTypeProperties> dTypes);
}

