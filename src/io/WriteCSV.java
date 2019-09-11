package io;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import org.apache.log4j.Logger;

/**
 * This class is used for writing to a CSV file. The class allows to manipulate
 * the fields to write to a CSV file by keeping, modifying or droping some
 * fields.
 * 
 * To use this class:
 * 1. A object must be created by specifying column names and file to write.
 * 2. If more headers want to be added use addHeader.
 * 3. Call the methods: drop, keep, reorderBy* to drop and reorder fields.
 * 4. Call fix headers if spaces are to be replaced with _ and put all headers
 * in lower case.
 * 5. Call finishedHeaders to create fields buffer and reordering arrays
 * (Perhaps the method should be called finishedConfiguration).
 * 6. Call writeHeaders.
 * 7. Do 1 to number of rows:
 *  - Add fields by calling addFields. As many fields must be added per row as
 *    there are headers.
 *  - Call replaceNulls, if nulls are to be replaced.
 *  - Call flushLine if done with this row.
 * 
 * 8. Call close.
 * 
 *
 * @author diego_paez
 */
public class WriteCSV {

    /**
     * Buffered Writer that is used to stored buffered data and then write a
     * whole buffer to file.
     */
    protected BufferedWriter fileWriter;

    /**
     * Name of the file that is going to be used to write to a file.
     */
    protected File fileToWrite;

    /**
     * CSV fields that will be written to file when flushed.
     */
    protected String[] fields;

    /**
     * Mapping of fields to a new ordering. For example if a field is at
     * position 7, then newOrder[7] will specify the new position for this
     * field, for example, position 3. In which case newOrder[7] = 3.
     */
    protected int[] newOrder;

    /**
     * Array used to specify which fields to keep. For example if fields: 0 1,
     * and 4 are specified, then field 2 and 3 will be dropped.
     */
    protected ArrayList<Integer> fieldsToWrite;

    /**
     * When adding info this variable serves to signal what is the current field
     * that will be added next.
     */
    protected int curField;

    /**
     * Number of columns that will be processed by this file. This includes all
     * columns including those that will not be written to the file.
     */
    protected int numColumns;
    
    /**
     * A String that specifies a reordering. For example if there are 5 fields,
     * then 4,0-3, specifies that the first position should be occupied by
     * field 4, and all other field must move one space. Indices start at 0.
     */
    protected String reorderString;
    
    /**
     * Used to specify a reordering by giving the name of variables that will
     * be swapped with thos of reorderSwap2.
     */
    protected String[] reorderSwap1;
    
    /**
     * Used to specify a reordering by giving the name of variables that will
     * be swapped with thos of reorderSwap1.
     */
    protected String[] reorderSwap2;
    
    /**
     * Map from name of headers to the index they occupy.
     */
    protected LinkedHashMap<String, Integer> headerMap;

    /**
     * Buffer Size
     */
    protected final int BUFFERSIZE = 4096; // Windows block size = 4096 bytes
    
    /**
     * Log4j logger.
     */
    private static final Logger LOG = Logger.getLogger(WriteCSV.class);

    /**
     * All columms within colNames will be added to headers map in memory, later
     * on they may be filtered.
     *
     * @param colNames Names of columns that will be added to be processed by
     * this class.
     * @param fileToWrite File to write to.
     * @throws java.io.IOException
     */
    public WriteCSV(List<String> colNames, File fileToWrite) throws IOException {

        // Construct header map.
        int j = 0;
        headerMap = new LinkedHashMap<>();
        if(colNames != null){
            for (String s : colNames) {
                headerMap.put(s, j++);
            }
        }

        numColumns = headerMap.size();

        this.fileToWrite = fileToWrite;
        reorderString = "";

        fileWriter = new BufferedWriter(new FileWriter(fileToWrite), BUFFERSIZE);
    }
    
    public WriteCSV(File fileToWrite)throws IOException{
        this(null,fileToWrite);
    }

    /**
     * Reorder based on the String given. For example if 2 variables exist then
     * the normal order would be "0,1" and to swap them the String "1,0" would be
     * given. A reordering of 5 variables might be: 4,1-3,0. Which means that
     * the variable with index 4 goes to place 0, 1 goes to 1, 2 to 2, 3 to 3
     * and 4 to 0.
     *
     * Reordering does not occur inmediately but rather when all variables have
     * been defined, i.e., when the method finishedHeaders is called. Thus
     * must be called before method finishedHeaders.
     *
     * @param places String with format to switch places.
     */
    public void reorderByIndices(String places) {
        reorderString = places;
    }

    /**
     * The name of the variables given in the swap1 array of Strings are swaped
     * in place for those in swap2.
     * 
     * @param swap1 Variables in that will be swaped with array swap2.
     * @param swap2 Variables in that will be swaped with array swap1.
     */
    public void reorderBySwapping(String[] swap1, String[] swap2) {
        reorderSwap1 = swap1;
        reorderSwap2 = swap2;
    }

    /**
     * The name of the variables given in the swap1 array of Strings are swaped
     * in place for those in swap2.
     * 
     * @param swap1 Variables in that will be swaped with array swap2.
     * @param swap2 Variables in that will be swaped with array swap1.
     */
    protected void reorderBySwapingInt(String[] swap1, String[] swap2) {
        Integer index1, index2;
        if (swap1.length != swap2.length) {
            System.out.println("Los tamanios para hacer swap no coinciden.");
            return;
        }

        for (int i = 0; i < swap1.length; i++) {

            // find index 1
            switch (swap1[i].toLowerCase()) {
                case "first":
                    index1 = 0;
                    break;
                case "last":
                    index1 = headerMap.size() - 1;
                    break;
                default:
                    index1 = headerMap.get(swap1[i]);
                    break;
            }

            // find index 2
            switch (swap2[i].toLowerCase()) {
                case "first":
                    index2 = 0;
                    break;
                case "last":
                    index2 = headerMap.size() - 1;
                    break;
                default:
                    index2 = headerMap.get(swap2[i]);
                    break;
            }

            if (index1 == null || index2 == null) {
                System.out.println("La variable con la cual hacer swap no fue hallada");
                continue;
            }

            newOrder[index1] = index2;
            newOrder[index2] = index1;
        }
    }

    /**
     * Reorder based on the String given. For example if 2 variables exist then
     * the normal order would be "0,1" and to swap them the String "1,0" would be
     * given. A reordering of 5 variables might be: 4,1-3,0. Which means that
     * the variable with index 4 goes to place 0, 1 goes to 1, 2 to 2, 3 to 3
     * and 4 to 0.
     *
     * @param places String with format to switch places.
     */
    protected void reorderByIndicesInt(String places) {
        // Start with a try in case there are format errors
        try {
            // Some auxiliary variables
            // auxOrder stores the order as given in the string
            int[] auxOrder = new int[numColumns];

            // split the string by commas
            String[] splitCommas = places.split(",");
            String[] splitHyphen;

            int j = 0;
            for (String splitComma : splitCommas) {
                // for each csv
                // if hyphen is contained
                if (splitComma.contains("-")) {
                    // split by hyphen
                    splitHyphen = splitComma.split("-");
                    // create numbers form start of hyphen till end of hyphen
                    int st = Integer.parseInt(splitHyphen[0]);
                    int end = Integer.parseInt(splitHyphen[1]);
                    for (int k = st; k <= end; k++) {
                        auxOrder[j] = k;
                        j++;
                    }
                } else {
                    // if no hyphen contained just assign the number
                    auxOrder[j] = Integer.parseInt(splitComma);
                    j++;
                }
            }

            // Flip the indices with the values
            for (int i = 0; i < numColumns; i++) {
                newOrder[auxOrder[i]] = i;
            }
        } catch (NumberFormatException ex) {
            System.out.println("Formato incorrecto");
        }
    }

    /**
     * Only writes to file the specified fields. Right now does not allow
     * to be used with drop.
     *
     * @param fieldsToKeep Indices of fields to keep.
     */
    public void keep(int[] fieldsToKeep) {
        fieldsToWrite = new ArrayList<>();
        for(int i = 0; i < fieldsToKeep.length; i++){
            fieldsToWrite.add(fieldsToKeep[i]);
        }
        Collections.sort(fieldsToWrite);
    }

    /**
     * Only writes to file the specified fields.
     *
     * @param fieldsToKeep Variable names of fields to keep.
     */
    public void keep(String[] fieldsToKeep) {
        keep(getIndices(fieldsToKeep));
    }

    /**
     * Remove the specified fields. Right now does not allow to be used with 
     * keep.
     *
     * @param fieldsToDrop The specified fields to drop.
     */
    public void drop(int[] fieldsToDrop) {
        fieldsToWrite = new ArrayList<>();
        Arrays.sort(fieldsToDrop);
        
        int j = 0;
        for(int i = 0; i < numColumns; i++){
            if(j < fieldsToDrop.length){
                if(fieldsToDrop[j] != i){
                    fieldsToWrite.add(i);
                } else{
                    j++;
                }
            } else{
                fieldsToWrite.add(i);
            }
        }
    }

    /**
     * Remove the specified fields. Right now does not allow to be used with 
     * keep.
     *
     * @param fieldsToDrop The specified fields to drop.
     */
    public void drop(String[] fieldsToDrop) {
        drop(getIndices(fieldsToDrop));
    }
    
    /**
     * Return array with indices for a given set of variables (as strings), 
     * names not found will not be included.
     * @param fieldsNames Names of variables to search.
     * @return Integer array with variable's names.
     */
    private int[] getIndices(String[] fieldsNames){
        ArrayList<Integer> fieldsInt = new ArrayList(fieldsNames.length);

        Integer found;
        for (String name : fieldsNames) {
            found = headerMap.get(name);
            if (found == null) {
                System.out.println("La columna " + name + " no fue hallada," + "esta columna no va a ser incluida.");
            } else {
                fieldsInt.add(found);
            }
        }

        int[] auxArr = new int[fieldsInt.size()];
        for(int i = 0; i < fieldsInt.size();i++){
            auxArr[i] = fieldsInt.get(i);
        }
        
        return auxArr;
    }

    /**
     * Add field to current row.
     * @param strToAdd Value to add.
     */
    public void addField(String strToAdd) {
        fields[newOrder[curField++]] = "\"" + strToAdd + "\"";
    }

    /**
     * Set field at given position for current row.
     * @param strRepl Value to set.
     * @param position Position at which to set.
     */
    public void setField(String strRepl, int position) {
        fields[newOrder[position]] = "\"" + strRepl + "\"";
    }

    /**
     * Add all fields given. Assumes array given is same length than number
     * of columns.
     * @param fields Array of fields to add. 
     */
    public void addAllFields(String[] fields) {

        if (fields.length != numColumns) {
            System.out.println("Can't add strings, array is of different length"
                    + "than fields");
        }
        for (int i = 0; i < numColumns; i++) {
            this.fields[newOrder[i]] = fields[i];
        }
    }

    /**
     * Replace fields between stIndex inclusive and endIndex exclusive
     *
     * @param stIndex
     * @param endIndex
     * @param replace
     */
    public void replaceNulls(int stIndex, int endIndex, String replace) {
        for (int i = stIndex; i < endIndex; i++) {
            if (fields[newOrder[i]] == null) {
                fields[newOrder[i]] = "\"" + replace + "\"";
            }

        }
    }

    /**
     * Current fields are written with Buffered Writer, i.e., they may be held
     * in memory until a full block is available to be written to disk.
     */
    public void flushLine() {

        StringBuilder strLine = new StringBuilder();

        if (fieldsToWrite == null) {
            for (String field : fields) {
                strLine.append(field).append(",");
            }
        } else {
            for (int i = 0; i < fieldsToWrite.size(); i++) {
                strLine.append(fields[fieldsToWrite.get(i)]).append(",");
            }
        }

        // Delete last comma
        strLine.deleteCharAt(strLine.length() - 1);
        // Add windows line break
        strLine.append("\r\n");

        try {
            fileWriter.write(strLine.toString());
        } catch (IOException ex) {
            LOG.error("Error while trying to write a CSV line to file: " +
                    strLine.toString(),ex);
        }

        // Reset fields
        curField = 0;

    }

    /**
     * Flushes remaining info in memory & closes file handle.
     */
    public void close() {
        try {
            fileWriter.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    /**
     * Get the path to the file.
     *
     * @return
     */
    public String getPath() {
        return fileToWrite.getPath();
    }

    /**
     * Get the name of the file without path.
     *
     * @return
     */
    public String getName() {
        return fileToWrite.getName();
    }

    /**
     * Get number of columns.
     * @return 
     */
    public int getNumColumns() {
        return numColumns;
    }

    /**
     * Get the name of the file without path and extension.
     *
     * @return
     */
    public String getNameWithoutExtension() {
        String name = fileToWrite.getName();
        return name.substring(0, name.length() - 4);
    }

    /**
     * Fix Headers by removing spaces and setting everything in lowercase.
     */
    public void fixHeaders() {
        // Get headers
        ArrayList<String> headers = new ArrayList<>(headerMap.keySet());
        LinkedHashMap<String, Integer> replace = new LinkedHashMap<>();
        int i = 0;
        String aux;

        for (String s : headers) {
            //
            aux = s.replace(' ', '_');
            aux = aux.toLowerCase();

            replace.put(aux, i++);
        }
        headerMap = replace;
    }

    /**
     * Add new header.
     *
     * @param header
     */
    public void addHeader(String header) {
        headerMap.put(header, numColumns++);
    }

    /**
     * These method is used to signal that all headers have been added and also
     * that columns that will be dropped, kept or reorder have been specified.
     */
    public void finishedHeaders() {

        newOrder = new int[numColumns];
        fields = new String[numColumns];

        for (int i = 0; i < numColumns; i++) {
            newOrder[i] = i;
        }

        // Reorder by indices if found
        if (!reorderString.equals("")) {
            reorderByIndicesInt(reorderString);
        }

        // Reorder by swap
        if (reorderSwap1 != null && reorderSwap2 != null) {
            reorderBySwapingInt(reorderSwap1, reorderSwap2);
        }
    }

    /**
     * Writes registered headers.
     */
    public void writeHeaders() {
        // Get headers
        ArrayList<String> headers = new ArrayList<>(headerMap.keySet());

        // Put every header inside fields array
        int i = 0;
        for (String s : headers) {
            fields[newOrder[i++]] = s;
        }

        // flush the names
        flushLine();
    }

    /**
     * Return file to write.
     *
     * @return
     */
    public File getFileToWrite() {
        return fileToWrite;
    }

    /**
     * Get a File object which has the same path as the file given and the name
     * given with csv extension.
     *
     * @param fileToRead
     * @param nameWrite
     * @return
     */
    public static File initializeWrite(File fileToRead, String nameWrite) {
        String path = fileToRead.getParentFile().getPath();
        String fileName = path + File.separator + nameWrite + ".csv";

        return new File(fileName);
    }

    /**
     * Create a WriteCSV object that will have the same headers as those found
     * in the variable csvABTRead, and that will have the same file name except
     * for the given suffix.
     * @param inputFile File from which to extract the headers' names, and the
     * file name.
     * @param suffix Suffix to add to file name to change name.
     * @return
     * @throws java.io.IOException
     */
    public static WriteCSV startReadWrite(ReadCSV inputFile,
            String suffix) throws IOException {

        // Write object
        String firstKeepName = inputFile.getReadFileName() + suffix;
        File readFile = new File(inputFile.getReadFileName());
        File fileToWrite = WriteCSV.initializeWrite(readFile, firstKeepName);
        return new WriteCSV(inputFile.getColumnNames(), fileToWrite);
    }
}
