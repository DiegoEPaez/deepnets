package io;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

/**
 * The use of this class is to read csv files by using the apache commons csv
 * parser. Particularly the class has methods for reading headers as well as
 * csv rows one at a time, it is up to the user to decide what to do with
 * the headers and the rows. The class assumes the file has header and must
 * be read using method readHeaders, before reading rows of data that are not
 * headers.
 * 
 * The apache parser is particularly useful because the general case
 * requires the use of a grammar (as specified in rfc 4180), and apache
 * takes care of that.
 * @author diego_paez
 */
public class ReadCSV {

    /**
     * File which contains the CSV data.
     */
    protected File fileToRead;
    
    /**
     * Names of the columns (headers).
     */
    protected ArrayList<String> columnNames;
    
    /**
     * Array with fields contained in each row that is read.
     */
    protected ArrayList<String> readFields;
    
    /**
     * File Reader used to read csv file.
     */
    protected FileReader in;
    
    /**
     * Apache CSV Parser used to parse the file.
     */
    protected CSVParser parser;
    
    /**
     * Map from the header's names to the place they occupy per column. The 
     * class in its current state only requires an array with the header's names
     * but the apache csv package, returns a Map. Despite the class only
     * requires header's names, the map could be useful in further improvements
     * to this class.
     */
    protected LinkedHashMap<String, Integer> headerMap;
    
    /**
     * Flag to signal whether the first row is being read. The use of this 
     * variable is to instantiate the CSVIterator when it is the first row.
     */
    protected boolean isFirst = true;
    
    /**
     * Iterator through all CSV Records.
     */
    protected Iterator<CSVRecord> iterator = null;

    /**
     * Constructor with CSV file.
     * @param file File which contains csv data.
     */
    public ReadCSV(File file) {
        fileToRead = file;
        try {
            //lineReader = new BufferedReader(new FileReader(file), BUFFERSIZE);
            in = new FileReader(file);
            CSVFormat format = CSVFormat.EXCEL.withHeader();
            parser = format.parse(in);
        } catch (IOException ex) {
            System.err.println(ex.getMessage());
            ex.printStackTrace();
        }
    }

    /**
     * Read headers of the CSV (assumes no BOM). Must be called before reading
     * rows.
     */
    public void readHeaders() {

        // Check for UTF-8 BOM
        /*   byte[] checkBom = headers.getBytes();
        if(checkBom.length >= 3 && checkBom[0] == (byte)0xEF &&
        checkBom[1] == (byte)0xBB && checkBom[2] == (byte)0xBF) // right knwo just delete BOM for UTF-8
        headers = headers.substring(1);
         */
        headerMap = (LinkedHashMap<String, Integer>) parser.getHeaderMap();

        columnNames = new ArrayList<>();
        for (String s : headerMap.keySet()) {
            columnNames.add(s.toLowerCase());
        }

    }

    /**
     * Return column names (headers).
     * @return Array with headers read.
     */
    public ArrayList<String> getColumnNames() {
        return columnNames;
    }

    /**
     * Return current fields (last row read).
     * @return Array with fields of last row read.
     */
    public ArrayList<String> getFields() {
        return readFields;
    }

    /**
     * Test if there is a next record to read. If that is the case put it into
     * the fields array.
     * @return Boolean whether there is a next row or not.
     */
    public boolean next() {

        if (isFirst) {
            iterator = parser.iterator();
            isFirst = false;
        }

        if (iterator.hasNext()) {

            readFields = new ArrayList<>();
            CSVRecord record = iterator.next();
            if (headerMap == null) {
                System.out.println("Header has not been read!");
            }

            for (String s : headerMap.keySet()) {
                readFields.add(record.get(s));
            }
            return true;

        }

        return false;
    }

    /**
     * Get field in given position.
     * @param pos Position of field to return.
     * @return Field in given pos.
     */
    public String get(int pos) {
        if (readFields.size() - 1 < pos || pos < 0) {
            return null;
        } else {
            return readFields.get(pos);
        }
    }

    /**
     * Get the file name without the path and extension from a File object (assumes
     * file has an extension of 3 characters plus period).
     * @return File name without the path and extension.
     */
    public String getReadFileName() {
        String name = fileToRead.getName();
        return name.substring(0, name.length() - 4);
    }

    /**
     * Size of the fields array (= number of columns).
     * @return Number of columns.
     */
    public int sizeFields() {
        return readFields.size();
    }

    /**
     * Close file reader.
     */
    public void close() {
        try {
            in.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
