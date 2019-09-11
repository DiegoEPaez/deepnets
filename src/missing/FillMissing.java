package missing;

import types.DataTypeM;
import types.DataTypeProperties;
import types.FindType;
import io.ReadCSV;
import io.WriteCSV;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author diego_paez
 */
public class FillMissing {

    private CategoricalMissingRule catRule;
    private NumericMissingRule numRule;
    private String suffix;

    public FillMissing(CategoricalMissingRule catRule, NumericMissingRule numRule,
            String suffix) {
        this.catRule = catRule;
        this.numRule = numRule;
        this.suffix = suffix;
    }
    
    public File completeMissing(String file,FindType types) throws IOException{

        ArrayList<DataTypeProperties> dTypes = types.varMap;

        // 1 find if passing is required if that is the case do so
        firstPass(file,dTypes);

        // 2 fill data with missing values
        return secondPass(file,dTypes);
    }

    /**
     * Applies 2nd pass directly, i.e., uses previously calculated values
     * @param file
     * @param types
     * @return
     * @throws IOException
     */
    public File completeMissingNC(String file, FindType types) throws IOException{
        ArrayList<DataTypeProperties> dTypes = types.varMap;

        // 2 fill data with missing values
        return secondPass(file,dTypes);
    }

    private void firstPass(String file, ArrayList<DataTypeProperties> dTypes){
        if(catRule.isPassRequired() || numRule.isPassRequired()){

            ReadCSV readCSV = new ReadCSV(new File(file));
            readCSV.readHeaders();
            ArrayList<String> fields;

            int k = 0;
            String val;
            while (readCSV.next()) {
                fields = readCSV.getFields();

                for(int i = 0; i < fields.size(); i++){
                    val = readCSV.get(i);

                    if(dTypes.get(i).getType() == DataTypeM.NUMERIC){
                        if(MissingValuesList.isMissingValue(val)) // missing
                            numRule.advanceVarIndex();
                        else
                            numRule.addData(Double.parseDouble(val));
                    } else{
                        if(MissingValuesList.isMissingValue(val))
                            catRule.advanceVarIndex();
                        else
                            catRule.addData(val);
                    }
                }
                k++;
            }
            readCSV.close();
        }

        numRule.passFinished();
        catRule.passFinished();
    }

    public File secondPass(String file,
            ArrayList<DataTypeProperties> dTypes) throws IOException{
        File oFile = new File(file);
        ReadCSV readCSV = new ReadCSV(oFile);
        
        readCSV.readHeaders();

        WriteCSV writeCSV = WriteCSV.startReadWrite(readCSV,suffix);
        writeCSV.finishedHeaders();
        writeCSV.writeHeaders();
        ArrayList<String> fields;

        int k = 0;
        String val;
        int numVarC, catVarC;
        while (readCSV.next()) {
            fields = readCSV.getFields();
            numVarC = 0;
            catVarC = 0;
            for(int i = 0; i < fields.size(); i++){
                val = readCSV.get(i);


                if(dTypes.get(i).getType() == DataTypeM.NUMERIC){
                    if(val == null || MissingValuesList.isMissingValue(val)) // missing
                        writeCSV.addField(String.valueOf(numRule.getReplaceValue(numVarC)));
                    else
                        writeCSV.addField(val);
                    numVarC++;
                } else{
                    if(val == null || MissingValuesList.isMissingValue(val))
                        writeCSV.addField(String.valueOf(catRule.getReplaceValue(catVarC)));
                    else
                        writeCSV.addField(val);
                    catVarC++;
                }
                
            }
            writeCSV.flushLine();
            k++;
        }
        readCSV.close();

        writeCSV.close();

        return writeCSV.getFileToWrite();
    }
}
