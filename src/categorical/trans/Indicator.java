package categorical.trans;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Set;
import types.DataTypeM;
import types.DataTypeProperties;

/**
 * Creates an indicator variable for number of classes - 1.
 * @author diego_paez
 */
public class Indicator implements TransformCategoricalRule {

    // Contains a serial assignment of each class for instance if classes are
    // A, B, C then strMap = {A -> 0}, {B -> 1}, {C -> 2}
    LinkedHashMap<String,Integer>[] strMap;
    private int curVar;

    /**
     * Initialize Indicator Transformation Rule.
     * @param strA Contains all classes for each categorical var.
     */
    public Indicator(Set<String>[] strA) {
        strMap = new LinkedHashMap[strA.length];

        int j;
        for(int i = 0; i < strA.length; i++){
            j = 0;
            strMap[i] = new LinkedHashMap();
            for(String str: strA[i]){
                strMap[i].put(str, j);
                j++;
            }
        }
    }

    /**
     * Replace a given string using current variable classes, advance to
     * next variable.
     * @param str
     * @return
     */
    @Override
    public String[] replace(String str) {
        int code = strMap[curVar].get(str);
        int size = strMap[curVar].keySet().size();

        String aux;
        String[] replace = new String[size - 1];

        for(int i = 0; i < size - 1; i++){
            if(code == i)
                aux = "1";
            else{
                aux = "0";
            }
            replace[i] = aux;
        }
        curVar = (curVar + 1) % strMap.length;
        return replace;
    }

    /**
     * Generates headers for the new transformed categorical variables given
     * the old ones.
     * @param colNames Original column names
     * @param dTypes Data Types (numeric or categorical) for the original column names
     * @return New headers for transformed data varaibe
     */
    @Override
    public ArrayList<String> generateHeaders(ArrayList<String> colNames,
            ArrayList<DataTypeProperties> dTypes) {
        ArrayList<String> newHeaders = new ArrayList();
        int size;
        int k = 0;
        for(int i = 0; i < colNames.size(); i++){
            if(dTypes.get(i).getType() == DataTypeM.NUMERIC){
                newHeaders.add(colNames.get(i));
            } else{
                size = strMap[k].keySet().size();
                for(int j = 0; j < size - 1; j++){
                    newHeaders.add(colNames.get(i) + "_" + j);
                }
                k++;
            }
        }

        return newHeaders;
    }
}