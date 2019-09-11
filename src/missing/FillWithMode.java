package missing;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 *
 * @author diego_paez
 */
public class FillWithMode implements CategoricalMissingRule{
    private LinkedHashMap<String,Integer>[] strMap;
    private String[] modes;
    private int currVar;

    public FillWithMode(int numVars) {
        strMap = new LinkedHashMap[numVars];

        for(int i = 0 ; i < numVars; i++){
            strMap[i] = new LinkedHashMap<>();
        }

        modes = new String[numVars];
    }

    @Override
    public boolean isPassRequired() {
        return true;
    }

    public LinkedHashMap<String, Integer>[] getStrMap() {
        return strMap;
    }

    public LinkedHashSet<String>[] getStrSet() {
        LinkedHashSet<String>[] set = new LinkedHashSet[strMap.length];
        for(int i = 0; i < strMap.length; i++){
            set[i] = new LinkedHashSet<>();
            set[i].addAll(strMap[i].keySet());
        }
        return set;
    }


    @Override
    public void addData(String data) {
        if(currVar == strMap.length)
            currVar = 0;

        Integer count;
        if((count = strMap[currVar].get(data)) == null){
            strMap[currVar].put(data, 1);
        } else{
            strMap[currVar].put(data, count + 1);
        }
        currVar++;
    }

    @Override
    public String getReplaceValue(int numVar) {
        return modes[numVar];
    }

    @Override
    public void advanceVarIndex() {
        currVar = (currVar + 1) % strMap.length;
    }

    @Override
    public void passFinished() {
        Set<String> values;
        Integer maxInt;
        Integer temp;
        String maxStr;
        for(int i = 0 ; i < strMap.length; i++){
            values = strMap[i].keySet();

            maxStr = "";
            maxInt = Integer.MIN_VALUE;
            for(String str: values){
                temp = strMap[i].get(str);
                if(temp > maxInt){
                    maxStr = str;
                    maxInt = temp;
                }
            }
            modes[i] = maxStr;
        }
    }
}
