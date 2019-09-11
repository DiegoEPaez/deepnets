package core;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import tensor.DoubleTensor;
import wregul.WeightRegularization;
import layer.Layer;
import layer.activation.SoftMaxLayer;
import layer.weight.WeightLayer;
import loss.CrossEntropy;
import loss.LossFunction;
import loss.WeightedCrossEntropy;
import org.apache.log4j.Logger;
import tensor.TensorFunctions;

/**
 * This class contains all information related to the structure & functionality
 * of a feed forward neural net.
 * 
 * TODO: Add Batch Normalization, should probably be added as a layer
 * see deeplearning4j
 * @author diego_paez
 */
public class NeuralNetModel {

    /**
     * Array of layers for a feed forward neural net.
     */
    public ArrayList<Layer> layers;
    
    /**
     * Loss Function (this is the function to be optimized)
     */
    public LossFunction loss;

    /**
     * Regularization Function for the case of adding a penalty to cost function
     * if weights are too big
     */
    public WeightRegularization wregul;

    /**
     * Flag to know whether a test data is going to run (important for dropout)
     * and RReLU.
     */
    public boolean testRunning;

    /**
     * Neural net weights.
     */
    public double[] theta;
    
    /**
     * Neural net gradient of weights.
     */
    public double[] thetaGrad;

    /**
     * Training set for neural net, containing independent variables.
     */
    public DoubleTensor Xtrain;
    
    /**
     * Training set for neural net, containing dependent variables.
     */
    public DoubleTensor ytrain;
    
    /**
     * Weights for each example in the training set.
     */
    public DoubleTensor yWeights;
    
    private static final Logger LOG = Logger.getLogger(NeuralNetModel.class);

    /**
     * Create a neural net model with specified loss function.
     * @param loss 
     */
    public NeuralNetModel(LossFunction loss) {
        this.loss = loss;
        layers = new ArrayList<>();
    }

    /**
     * Add layer to this net. The layer is added to the last place of an array of
     * layers. Thus, last layer added  generates output of the net, and first
     * layer added takes input.
     * @param layer Layer to add.
     * @return This model so as to chain addLayers: model.addLayer(A).addLayer(B)...
     */
    public NeuralNetModel addLayer(Layer layer){
        layers.add(layer);
        return this;
    }

    /**
     * Set train inputs of the neural net.
     * @param X Independent variables, which is a tensor with any number of
     * dimensions (this depends also on the structure of the net), last dimension
     * must be the number of examples. For example for the MNIST data set the
     * number of rows of an image is stored in the first dim, the number of
     * columns (pixels) of an image is stored in the second dim, and the number
     * of images read is stored in the third and last dimension.
     * @param y Target variable, usually a 1 dimensional tensor with size =
     * number of examples (but can have more dims depending on loss function).
     * For the case of CrossEntropy loss function a 1 dimensional Tensor is fine
     * since the function "index" is used to expand each entry of the vector
     * to a 2 dim. Tensor.
     * @param yWeights Weights for the examples (i.e., to give more or less
     * importance to each example). Either null or a 1 dimensional tensor with
     * size = number of examples.
     */
    public void setTrainInputs(DoubleTensor X, DoubleTensor y,
            DoubleTensor yWeights){
        this.Xtrain = X;
        this.ytrain = y;
        this.yWeights = yWeights;
        setInputs(X.dims);
    }

    /**
     * Auxiliary method which sets the dimensions of the layers, using the size
     * of X, without taking into account examples. The dimension of the layers
     * are calculated so that parameters can be randomly initialized.
     * 
     * The reason for leaving the number of examples out of this method is that
     * when optimizing different batch sizes may be run through the net so that
     * dimension should be fixed.
     * With the input dimension of each layer the output dimensions can be
     * calculated which in turn are the input to the next layer. For the case
     * of activation layers #inputs = #outputs (WOE). For the case of weight
     * layers as well as subsampling layers the number of outputs differ.
     * @param Xdims
     */
    private void setInputs(int[] Xdims){
        int [] dimsIn = new int[Xdims.length - 1];
        System.arraycopy(Xdims, 0, dimsIn, 0, dimsIn.length);

        // For each layer set inputs size and get output sizes.
        for(Layer l: layers){
            l.setDimsOfInputsWOE(dimsIn);
            dimsIn = l.getDimsOfOutputsWOE();
        }
    }

    /**
     * Initializes the parameters of each layer by calling the initialization
     * method of each layer. Some layers might not have any parameters to
     * initialize (but that is up to each layer).
     * 
     * Once all parameters have been initialized a copy of the parameters
     * is made in a local array that contains all parameters of all layers.
     * 
     * The need for an array with all parameters is that the optimization method
     * requires a single array of parameters. An array of pointers to the
     * array within each layer could also be saved, the trouble is the optimization
     * method would have to admit this kind of input. Admitting this kind of
     * input could be complicated especially for methods like AdamUpdate which
     * stores 2 cached arrays. If memory or processing were a heavy burden on
     * the net, an array of pointers could be stored, but that is not the case
     * from profiling run.
     */
    public void initWeights(){
        int totalParams = 0;
        WeightLayer wl;

        for(Layer l: layers){
            if(l instanceof WeightLayer){
                wl = (WeightLayer) l;
                wl.initParams();
                totalParams += wl.getNumberOfParams();
            }
        }

        // initialize the array with all parameters
        theta = new double[totalParams];
        thetaGrad = new double[totalParams];

        // copy current init
        copyTheta();
    }

    /**
     * Loads a set of weights into the neural net.
     * @param file File where weights are stored.
     */
    public void loadWeights(String file){
        ArrayList<Double> weights = new ArrayList<>();
        try {
            DataInputStream is;
            is = new DataInputStream(new FileInputStream(file));
            while(true){
                weights.add(is.readDouble());
            }
        } catch(EOFException e){
            LOG.info("Finished reading weights from file " + file);
        } catch(IOException e){
            LOG.error("Problem while trying to load weights",e);
        }
        Double[] result = new Double[weights.size()];
        double[] presult = new double[weights.size()];
        result = weights.toArray(result);
        for(int i = 0; i < result.length; i++){
            presult[i] = result[i];
        }

        setWeights(presult);
    }

    /**
     * Initialize arrays that will contain examples plus set the number of examples
     * as last dim. This is necessary because for example Stochastic Gradient
     * Descent uses batches of different sizes.
     * @param numExamples Number of Examples with which to initialize net.
     */
   private void initSpace(int numExamples){
        for(Layer l: layers){
            l.initSpaceInMemory(numExamples);
        }
    }

    /**
     * Loads a vector x of weights into each layer's array of weights, as well
     * as into the local array
     * @param x Weights to pass on to each layer.
     */
    public void setWeights(double[] x){
        WeightLayer wl;
        int k = 0, bSize, wSize;
        int totalParams = 0;
        for(Layer l: layers){
            if(l instanceof WeightLayer){
                wl = (WeightLayer) l;

                if(wl.weights == null){
                    wl.initParams();

                }
                totalParams += wl.getNumberOfParams();

                // copy bias
                bSize = wl.bias.size();
                for(int i = 0; i < bSize;i++){
                    wl.bias.data.setQuick(i, x[k]);
                    k++;
                }

                // copy non bias weights
                wSize = wl.weights.size();
                for(int i = 0; i < wSize;i++){
                    wl.weights.data.setQuick(i, x[k]);
                    k++;
                }
            }
        }

        // initialize the array with all parameters
        if(theta == null){
            theta = new double[totalParams];
            thetaGrad = new double[totalParams];
        }

        // copy current init
        copyTheta();
    }

    /**
     * Calculates the total number of parameters by adding the total number
     * of parameters of each layer
     * @return Total number of parameters of the whole net.
     */
    public int getNumberOfParams(){
        int total = 0;
        WeightLayer wl;
        for(Layer l: layers){
            if(l instanceof WeightLayer){
                wl = (WeightLayer) l;
                total += wl.getNumberOfParams();
            }
        }

        return total;
    }

    /**
     * Forward propagate a given Tensor X.
     * @param X input data.
     * @return Output of forward propagation (without loss function).
     */
    public DoubleTensor fProp(DoubleTensor X){
        // update num. of examples to X num. of examples
        // (if X is a batch it might have different sizes)
        initSpace(X.lastDim());
        
        // forward propagate by calling each layer's forward propagation algorithm
        DoubleTensor Z = X;
        for (int i = 0; i < layers.size(); i++){
            Layer layer = layers.get(i);
            Z = layer.fProp(Z, testRunning);
        }

        return Z;
    }

    /**
     * Calculate the total cost function. This is obtained by forward propagating
     * the neural net. Once the net is forward propagated the loss function is
     * applied over the output of the net. Also any regularization penalty is also
     * calculated and added.
     * @param X Input data.
     * @param y Labels of each example.
     * @param yWeights Weights to apply to each example.
     * @param lambdaAdjus Adjust the lambda penalty if a smaller set of data
     * was passed.
     * @return Cost of the function (error made by the net) 
     */
    public double cost(DoubleTensor X, DoubleTensor y, DoubleTensor yWeights,
             double lambdaAdjus){
        DoubleTensor output = fProp(X);
        double netCost = loss.eval(output, y, yWeights);
        double regulCost = 0.0;
        if(wregul != null)
            regulCost = wregul.eval(this,lambdaAdjus);
        return netCost + regulCost;
    }

    /**
     * Backpropagate the cost of the function without any regularization.
     * To backpropagate, each layer's backpropagation method is called, as well
     * as the method that updates the layer's parameter's.
     * @param nnOutput Output of the neural net from forward propagation.
     * @param y Actual training values of the target variable.
     * @param yWeights Weight to apply to each training example.
     */
    public void bPropWOR(DoubleTensor nnOutput, DoubleTensor y, DoubleTensor yWeights){
        int startLayer;
        // Auxiliar pointer of current weight layer
        WeightLayer wl;
        // chaingrad stores derivative of Loss Function with respect to
        // the output of prev. layer (input of current layer)
        DoubleTensor chainGrad;

        /* When the loss function is CrossEntropy or WeightedCrossEntropy
        and the output layer is SoftMax the chainGrad is calculated at once.
        The only reason to do this is a speed up of about 3 to 7 times faster
        than calculating separately SoftMax chaingrad + Cross Entropy chaingrad.
        */
        if(loss instanceof CrossEntropy && layers.get(layers.size() - 1) instanceof SoftMaxLayer){
            chainGrad = gradCE_Softmax(nnOutput, y);
            startLayer = layers.size() - 2;
        } else if(loss instanceof WeightedCrossEntropy && layers.get(layers.size() - 1) instanceof SoftMaxLayer){
            chainGrad = gradWCE_Softmax(nnOutput, y, yWeights);
            startLayer = layers.size() - 2;
        } else{
            chainGrad = loss.bProp(nnOutput, y, yWeights);
            startLayer = layers.size() - 1;
        }
        
        // For the rest of the layers update weights and backpropagate
        for(int i = startLayer; i >= 0; i-- ){
            
            if(layers.get(i) instanceof WeightLayer){
                wl = (WeightLayer) layers.get(i);
                wl.updateLayerWGrad(chainGrad);
            }
            // no need to backpropagate first layer (i > 0)
            if(i > 0)
                chainGrad = layers.get(i).bProp(chainGrad);
            
        }
    }

    /**
     * Backpropagate including regularization. Uses bPropWOR method plus wregul
     * update method to backpropagate.
     * @param y Values of target variable.
     * @param yWeights Weights for each target variable.
     * @param lambdaAdjus Lambda adjustement of the regularization parameter.
     */
    public void bProp(DoubleTensor y, DoubleTensor yWeights,
            double lambdaAdjus){
        bPropWOR(layers.get(layers.size() - 1).getOutput(), y, yWeights);
        if(wregul != null)
            wregul.updateWeights(this,lambdaAdjus);
        // copy gradient obtained by each layer into a single array; used by optimizer
        copyThetaGrad();
    }

    /**
     * Special case where it is best to calculate the gradient of Cross Entropy
     * at once when output unit is a Softmax unit. 
     *
     * calculates directly dJ(.) / dz_lm which equals (for the lth input, mth example)
     *
     * - [(Indicator(y_(m),l) - softmax(l,Z)) / (1 + smallConst /  softmax(y_(m),Z))]
     *
     * Where
     * indicator(y_(m),l) = 1 if y_(m) = l else = 0
     * softmax(l,Z) = exp(Z_lm) / sum(Z_jm,j=1...K) = U_lm
     *
     * Normally softmax(y_(m),Z) > smallConst and thus smallConst / softmax(y_(m),Z) = 0
     *
     * But it can occur that softmax(y_(m),Z) is really small and thus
     * smallConst /softmax(y_(m),Z) gets bigger.
     *
     * @param nnOut Output of neural net.
     * @param y Values of target variable.
     * @return Derivative of the loss function J, with repect to inputs of Softmax
     * function
     */
    public static DoubleTensor gradCE_Softmax(DoubleTensor nnOut, DoubleTensor y){
        DoubleTensor indicator, temp, indSC, temp2;

        // get indicator matrix for vector y
        indicator = y.index(nnOut.dims[0]);

        // temp = indicator.sub(ui);
        // temp = temp.mul(-1.0); // outer minus
        temp = indicator.rsubi(nnOut);

        // this part deals with the extra work required by adding the small constant.
        // Usually temp2 should equal a matrix of ones, but with really small
        // values of ui, it gives different results.
        indSC = nnOut.getLowerDim(0, y);
        temp2 = indSC.rdivi(TensorFunctions.smallConst).addi(1.0);

        return temp.diviLowerDimTensor(0,temp2);
    }

    /**
     * Special case where it is best to calculate the gradient of Cross Entropy
     * at once when output unit is a Softmax unit. If calculated appart softmax
     * gradient creates m = no. of examples matrices of n x n
     *
     * calculates directly dJ(.) / dzj_K
     * @param nnOut Output of neural net.
     * @param y Values of target variable.
     * @param weights Weights of the target var.
     * @return Derivative of the loss function J, with repect to inputs of Softmax
     * function
     */
    public static DoubleTensor gradWCE_Softmax(DoubleTensor nnOut, DoubleTensor y,
            DoubleTensor weights){
        DoubleTensor indicator, temp, indSC, temp2;

        // get indicator matrix for vector y
        indicator = y.index(nnOut.dims[0]);
        temp = indicator.rsubi(nnOut);
        temp.muliLowerDimTensor(0, weights);

        // this part deals with the extra work required by adding the small constant.
        // Usually temp2 should equal a matrix of ones, but with really small
        // values of ui, it gives different results.
        indSC = nnOut.getLowerDim(0, y);
        temp2 = indSC.rdivi(TensorFunctions.smallConst).addi(1.0);

        return temp.diviLowerDimTensor(0,temp2);
    }

    /**
     * Copies each of the weights of each layer into a single array.
     */
    private void copyTheta(){
        WeightLayer wl;
        int k = 0;
        for(Layer l: layers){
            if(l instanceof WeightLayer){
                wl = (WeightLayer) l;

                // copy bias
                for(int i = 0; i < wl.bias.size();i++){
                    theta[k] = wl.bias.data.getQuick(i);
                    k++;
                }

                // copy non bias weights
                for(int i = 0; i < wl.weights.size();i++){
                    theta[k] = wl.weights.data.getQuick(i);
                    k++;
                }
            }
        }
    }

    /**
     * Copy each of the gradients of each layer into a single array.
     */
    private void copyThetaGrad(){
        WeightLayer wl;
        int k = 0;
        for(Layer l: layers){
            if(l instanceof WeightLayer){
                wl = (WeightLayer) l;

                // copy bias
                for(int i = 0; i < wl.biasGrad.size();i++){
                    thetaGrad[k] = wl.biasGrad.data.getQuick(i);
                    k++;
                }

                // copy non bias weights
                for(int i = 0; i < wl.weightsGrad.size();i++){
                    thetaGrad[k] = wl.weightsGrad.data.getQuick(i);
                    k++;
                }
            }
        }
    }
}