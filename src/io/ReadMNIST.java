package io;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import org.apache.log4j.Logger;
import tensor.DoubleTensor;

/**
 * This class allows to read the MNIST data set. It also normalizes the data
 * (subtract min. value and divide by range, per example /column).
 * @author diego_paez
 */
public class ReadMNIST {

    /**
     * Array used to store digits which are either 0 or 1. This is used in case
     * only digits with 0 or 1 want to be read. Assumes labels are read first.
     */
    private static int[] indexBinDigits;
    
    /**
     * Logger for this class.
     */
    private static final Logger LOG = Logger.getLogger(ReadMNIST.class);

    /**
     * Loads data with MNIST format into a DoubleTensor with dimensions: number
     * of rows, number of columns and examples. This method allows to filter
     * the images by binary digits (0 or 1), if only these want to be read.
     * Also allows to read a limited amount of images in to memory.
     * Finally it also allows to read images with padding meaning images can
     * have 0 surrounding the image, so that a learning algo. can better train.
     * 
     * From MNIST doc:
     *
     * TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000803(2051) magic number
     * 0004     32 bit integer  60000            number of images
     * 0008     32 bit integer  28               number of rows
     * 0012     32 bit integer  28               number of columns
     * 0016     unsigned byte   ??               pixel
     * 0017     unsigned byte   ??               pixel
     * ........
     * xxxx     unsigned byte   ??               pixel
     * Pixels are organized row-wise. Pixel values are 0 to 255. 0 means
     * background (white), 255 means foreground (black).
     *
     * @param fileName Name of the file to read.
     * @param binaryDigits Whether to only read binary digits or not.
     * @param numToRead Number of images to read.
     * @param pad Number of zeros to be placed in the edges (note that the
     * number of total rows will be rows + 2 * pad, same applies to columns).
     * @return Tensor with MNIST Images.
     */
    public static DoubleTensor loadMNISTImages(String fileName, boolean binaryDigits
            , int numToRead, int pad){
        BufferedInputStream br = null;
        byte[] buffer;
        int magic, numImages, numRows, numCols, size, index, nRPad, nCPad;
        double[] resData = null;
        int[] dims = new int[3];
        int[] indices = new int[3];

        try {
            
            // read into a buffered stream
            br = new BufferedInputStream(
                new FileInputStream(new File(fileName)));
            
            // Read magic number (first 4 bytes should equal 2051 in big endian)
            magic = readBigEndianInt(br);
            
            // if magic number is wrong exit
            if(magic != 2051){
                LOG.error("Magic number should be 2051, but read " + magic);
                System.exit(1);
            }

            // read number of images, rows and columns
            numImages = readBigEndianInt(br);
            numRows = readBigEndianInt(br);
            numCols = readBigEndianInt(br);
            
            // calculate number of rows and columns with padding
            nRPad = numRows + pad * 2;
            nCPad = numCols + pad * 2;
            
            // set first 2 dimensions to the number of rows and columns
            dims[0] = nRPad;
            dims[1] = nCPad;
            
            // total image size including padding
            int imgSize = dims[0] * dims[1];
            
            // create buffer to store images
            buffer = new byte[imgSize];

            // If using binary digits, assumes labels have been read and assigns same size
            if(binaryDigits){
                resData = new double[imgSize * indexBinDigits.length];
                dims[2] = indexBinDigits.length;
            } else {
                // assign to last dimension according to the number of images to read
                if(numToRead < 0 || numToRead >= numImages)
                    size = numImages;
                else
                    size = numToRead;
                
                dims[2] = size;

                resData = new double[imgSize * size];
            }

            byte nextPixel;
            int k = 0;
            int j = 0;
            
            // start reading
            while (br.read(buffer) != -1) {

                // Skip this row if binary flag is set and it is not a binary digit (or
                // exceeded indexBinDigits length
                if(binaryDigits && (k >= indexBinDigits.length || indexBinDigits[k] != j)){
                    j++;
                    continue;
                }

                // Skip if exceeded number of images to read
                if(numToRead > 0 && k >= numToRead){
                    break;
                }

                // flip info, since pixels are organized row-wise and first dim
                // in tensor are rows (meaning to store info we need to advance
                // first by rows)
                for(int iCP = 0, iC = 0; iCP < nCPad; iCP++){
                    for(int iRP = 0, iR = 0; iRP < nRPad; iRP++){
                        indices[0] = iCP;
                        indices[1] = iRP;
                        indices[2] = k;
                        index = DoubleTensor.indicesToNum(indices, dims);

                        // if within padding region set to 0
                        if(iRP < pad || iRP > nRPad - pad || iCP < pad || iCP > nCPad - pad){
                            resData[index] = 0.0;
                        } else{
                            // read next pixel which are by rows
                            nextPixel = buffer[iR + iC * numRows];

                            // switch rows and columns (we want to store data by columns,
                            // since first dim of tensor = rows, i.e. the first index
                            // that must advance are rows)
                            resData[index] =
                                    (double)(( (nextPixel & 0xFF)) / 255.0);
                        }
                        if(iRP >= pad && iRP <= nRPad - pad)
                            iR++;
                    }
                    if(iCP >= pad && iCP <= nRPad - pad)
                        iC++;
                }
                j++;
                k++;
            }
            
        } catch (IOException e) {
            System.out.println("err");
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        if(resData != null)
            return new DoubleTensor(resData, dims);
        else
            return null;
    }

    /**
     * Read labels with MNIST format into a Tensor. From the web page:
     * 
     * [offset] [type]          [value]          [description] 
     * 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
     * 0004     32 bit integer  60000            number of items 
     * 0008     unsigned byte   ??               label 
     * 0009     unsigned byte   ??               label 
     * ........ 
     * xxxx     unsigned byte   ??               label
     * The labels values are 0 to 9.
     * 
     * @param fileName Name of the file to read.
     * @param binaryDigits Whether to only read binary digits or not.
     * @param numToRead Number of labels to read.
     * @return Tensor with labels
     */
    public static DoubleTensor loadMNISTLabels(String fileName,
            boolean binaryDigits, int numToRead){
        BufferedInputStream br = null;
        byte[] buffer;
        int magic, numLabels;
        double[] resData = null;

        try {
            br = new BufferedInputStream(
                new FileInputStream(new File(fileName)));
            
            // Read magic number ( check MNIST file format)
            magic = readBigEndianInt(br);
            
            // If magic number is incorrect exit 
            if(magic != 2049){
                LOG.error("Magic number should be 2049, but read " + magic);
                System.exit(1);
            }

            numLabels = readBigEndianInt(br);
            buffer = new byte[numLabels];
            br.read(buffer);

            // read labels (if using binary labels filter out)
            if(binaryDigits){
                // Count binary labels ( 0 or 1 )
                int countDig = 0;
                for(int i = 0; i < numLabels; i++){
                    if(buffer[i] == 0 || buffer[i] == 1)
                        countDig++;

                    if(numToRead > 0 && countDig >= numToRead)
                        break;
                }

                // save labels in resMatrix
                resData = new double[countDig];
                indexBinDigits = new int[countDig];
                int k = 0;
                for(int i = 0; i < numLabels && k < countDig; i++){
                    if(buffer[i] == 0 || buffer[i] == 1){
                        indexBinDigits[k] = i;
                        resData[k] = (double)buffer[i];
                        k++;
                    }
                }
            } else{
                int size;

                if(numToRead < 0 || numToRead >= numLabels)
                    size = numLabels;
                else
                    size = numToRead;

                resData = new double[size];
                for(int i = 0; i < size; i++){
                    resData[i] = (double)buffer[i];
                }
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        if(resData != null)
            return new DoubleTensor(resData,new int[]{resData.length});
        else
            return null;
    }

    /**
     * Reads into memory the MNIST datasets, both training and test datasets, and
     * then normalizes them.
     * @param binaryDigits
     * @param numToRead
     * @param zeroPadSize
     * @return 
     */
    public static DoubleTensor[] loadMNIST(boolean binaryDigits, int numToRead,
            int zeroPadSize){

        // Load the training data
        DoubleTensor y = loadMNISTLabels("train-labels.idx1-ubyte",binaryDigits,numToRead);
        DoubleTensor X = loadMNISTImages("train-images.idx3-ubyte",binaryDigits,numToRead,zeroPadSize);

        // Randomly shuffle the data
        // DoubleMatrix[] res = MatrixUtil.randomizeColumns(X, y);
        // X = res[0];
        // y = res[1];

        ScaleData sData = new ScaleData();
        
        // We normalize the data, i.e.,moved from [min,max] range to [0,1] range
        // last dim = examples leave these fixed
        DoubleTensor trainX = sData.normalizei(2, X);
        DoubleTensor trainY = y;

        // Load the test data
        y = loadMNISTLabels("t10k-labels.idx1-ubyte",binaryDigits,numToRead);
        X = loadMNISTImages("t10k-images.idx3-ubyte",binaryDigits,numToRead,zeroPadSize);

        //Randomly shuffle the data ( not really necesary for test data)
        /*res = MatrixUtil.randomizeColumns(X, y,10);
        X = res[0];
        y = res[1];*/

        // Normalize test data, with training set calculated values (to keep
        // consistency).
        DoubleTensor testX = sData.normalizeiWithCalculated(2,X);
        DoubleTensor testY = y;
        
        // return normalized tensors.
        return new DoubleTensor[]{trainX,trainY,testX,testY};
    }

    /**
     * Read big endian ints. Normally bytes are in little endian format for
     * x86 processors, thus a special method is required for big endian ints,
     * where the biggest bits are first.
     * @param br Stream from which to read info.
     * @return Integer of 4 bytes with big endian format.
     * @throws IOException 
     */
    private static int readBigEndianInt(BufferedInputStream br) throws IOException{
        int retInt;
        
        // move first byte to the leftmost position
        retInt = br.read() << 24;
        
        // move second byte to the second leftmost position.
        retInt = retInt | (br.read() << 16);
        
        // move third byte to the third leftmost position
        retInt = retInt | (br.read() << 8);
        
        // read last byte and put it on the rightmos position
        retInt = retInt | br.read();

        return retInt;
    }
}