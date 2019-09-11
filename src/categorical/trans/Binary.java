package categorical.trans;

import types.DataTypeM;
import types.DataTypeProperties;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Set;

/**
 * Transform categorical variables into binary coding. Each class is assigned
 * a number from 0 to n. Once each class has a number assigned, the minimum number
 * of bits required to write n in binary is calculated. Finally each number corresponding
 * to a class is transformed to binary and each bit uses 1 column.
 * @author diego_paez
 */
public class Binary implements TransformCategoricalRule {

    // Contains a serial assignment of each class. For instance if classes are
    // A, B, C then strMap = {A -> 0}, {B -> 1}, {C -> 2}
    LinkedHashMap<String,Integer>[] strMap;
    private int curVar;

    /**
     * Initilize binary transformation rule.
     * @param strA Contains all classes for each categorical var.
     */
    public Binary(Set<String>[] strA) {
        strMap = new LinkedHashMap[strA.length];

        int j;
        // Create serial assigment of each class
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
     * Generates Headers for the new transformed categorical variables given
     * the old ones.
     * @param colNames Original column names
     * @param dTypes Data Types (numeric or categorical) for the original column names
     * @return New headers for transformed data varaibe
     */
    public ArrayList<String> generateHeaders(ArrayList<String> colNames,
            ArrayList<DataTypeProperties> dTypes){
        ArrayList<String> newHeaders = new ArrayList();
        int size, numBits;
        int k = 0;

        // Loop through each of the headers in the original file
        for(int i = 0; i < colNames.size(); i++){

            // if variable is numeric just add header as is
            if(dTypes.get(i).getType() == DataTypeM.NUMERIC){
                newHeaders.add(colNames.get(i));
            } else{ // for categorical variables get the number of bits to use and make as many headers
                size = strMap[k].keySet().size();
                numBits = getNumBits(size - 1);
                for(int j = 0; j < numBits; j++){
                    newHeaders.add(colNames.get(i) + "_" + j);
                }
                k++;
            }
        }

        return newHeaders;
    }

    /**
     * Count number of bits required to write the integer given.
     * @param size 
     * @return
     */
    private int getNumBits(int size){
        int numBits = 1;
        // first bit is always necessary
        size = size >> 1;
        while( size > 0){
            numBits++;
            size >>= 1;
        }

        return numBits;
    }

    /**
     * Replace a given string using current variable classes, advance to next
     * variable.
     * @param str
     * @return
     */
    @Override
    public String[] replace(String str) {
        int code = strMap[curVar].get(str);
        int size = strMap[curVar].keySet().size();
        int numBits = getNumBits(size - 1);

        // get binary representation of the numeric value
        String strCode = Integer.toBinaryString(code);

        String aux;
        char[] arr;

        String[] replace = new String[numBits];

        int j = 0;
        int k = 0;

        // create a string for each bit
        for(int i = numBits - 1; i >= 0; i--){
            if(strCode.length() < i + 1)
                aux = "0";
            else{
                arr = new char[]{strCode.charAt(k)};
                aux = new String(arr);
                k++;
            }
            replace[j] = aux;
            j++;
        }
        curVar = (curVar + 1) % strMap.length;
        return replace;
    }
}