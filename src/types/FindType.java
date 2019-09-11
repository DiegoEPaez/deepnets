package types;

import io.ReadCSV;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import missing.MissingValuesList;
import org.apache.commons.lang3.math.NumberUtils;

/**
 *
 * @author DP12577
 */
public class FindType {

    public ArrayList<DataTypeProperties> varMap;

    public FindType(){
        varMap = new ArrayList<>();
    }

    public int size(){
        return varMap.size();
    }

    public void checkRow(List<String> values){
        int i = 0;
        long len, len2;

        for(String s: values){
            if(MissingValuesList.isMissingValue(s)){
                i++;
                continue;
            }
            if(!NumberUtils.isNumber(s)){
                if(i == 3)
                    System.out.println("Err");
                varMap.get(i).setType(DataTypeM.CATEGORICAL);
                len = varMap.get(i).getLength();
                len2 = s.length();

                if(len <= len2)
                    varMap.get(i).setLength(len2);
            }
            i++;
        }
    }

    public DataTypeProperties get(int i){
        return varMap.get(i);
    }

    public void loopThruFile(String file){
        ReadCSV readCSV = new ReadCSV(new File(file));
        readCSV.readHeaders();
        ArrayList<String> columns = readCSV.getColumnNames();

        for(int i = 0; i < columns.size(); i++){
            varMap.add(new DataTypeProperties(DataTypeM.NUMERIC,0));
        }

        ArrayList<String> fields;

        while (readCSV.next()) {
            fields = readCSV.getFields();
            checkRow(fields);
        }
        readCSV.close();
    }

    public int numericCount(){
        int nCt = 0;
        for(int i = 0; i < varMap.size();i++){
            if(varMap.get(i).getType() == DataTypeM.NUMERIC)
                nCt++;
        }
        return nCt;
    }

    public int catCount(){
        int cCt = 0;
        for(int i = 0; i < varMap.size();i++){
            if(varMap.get(i).getType() == DataTypeM.CATEGORICAL)
                cCt++;
        }
        return cCt;
    }
}
