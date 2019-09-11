package main;

import core.NeuralNetFunction;
import core.NeuralNetGradient;
import io.ReadMNIST;
import layer.activation.ActivationLayer;
import layer.activation.SoftMaxLayer;
import layer.weight.InnerProductLayer;
import loss.CrossEntropy;
import core.NeuralNetModel;
import eval.EvalUtil;
import init.He;
import java.io.File;
import layer.activation.ReLU;
import layer.subsampling.MeanPooling2DLayer;
import layer.weight.Convolution2DLayer;
import optim.sgd.SGDInputs;
import optim.sgd.SGDOptim;
import optim.sgd.plot.PlotLFOptions;
import optim.sgd.plot.PlotLFType;
import optim.sgd.update.AdamUpdate;
import org.apache.log4j.Logger;
import tensor.DoubleTensor;

/**
 * @author diego_paez
 */
public class RunLeNet {

    private static final String SAVEFILE = "weights.dat";
    private static final Logger LOG = Logger.getLogger(RunLeNet.class);

    public static void main(String [] args){
        LOG.info("Running Neural Net Regression");
        LOG.info("Loading MNIST Data");
        DoubleTensor[] trainTest = ReadMNIST.loadMNIST(false,5000,0);

        DoubleTensor trainX = trainTest[0];
        DoubleTensor trainY = trainTest[1];
        DoubleTensor testX = trainTest[2];
        DoubleTensor testY = trainTest[3];

        NeuralNetModel model = new NeuralNetModel(new CrossEntropy());

        /*model.addLayer(new InnerProductLayer(new He(), 10))
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new SoftMaxLayer());*/

        // Variation of LeNet-5 actually uses original uses Sigmoid not ReLU
         // and for pooling uses weights 2, current mean pooling does not use weights
        // 98.56% best net so far... it hasn't been trained fully (epochs = 13)
        model.addLayer(new Convolution2DLayer(new He(), 20, 5, 5, 1, 1,
                Convolution2DLayer.ConvolveMethod.FFT))
             .addLayer(new MeanPooling2DLayer(2, 2))
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new Convolution2DLayer(new He(), 40, 5, 5, 1, 1,
                Convolution2DLayer.ConvolveMethod.FFT))
             .addLayer(new MeanPooling2DLayer(2, 2))
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new InnerProductLayer(new He(), 100)) 
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new InnerProductLayer(new He(), 10))
             .addLayer(new SoftMaxLayer());

        model.setTrainInputs(trainX, trainY, null);
        File testW = new File(SAVEFILE);
        if(testW.exists()){
            model.loadWeights(SAVEFILE);
        } else{
            model.initWeights();
        }
        
        NeuralNetFunction fun = new NeuralNetFunction(model);
        NeuralNetGradient grad = new NeuralNetGradient(model);
        
        // set optimization parameters
        SGDInputs inputs = new SGDInputs(model.theta,fun,grad,trainX,trainY);
        inputs.updater = new AdamUpdate(1E-3, 0.9, 0.999, model.theta.length);
        inputs.batchSize = 250;
        inputs.epochs = 40;
        inputs.annealEvery = 2;
        inputs.annealRate = 2;
        inputs.alwaysAnneal = false;
        inputs.plotLFOptions = new PlotLFOptions(PlotLFType.ALLBATCHES, 0, 0, 1000);
        inputs.saveWeights = true;
        inputs.file = SAVEFILE;
        SGDOptim optimizer = new SGDOptim();
        optimizer.optim(inputs);

        // Print out misclassified train
        int numMissClas = EvalUtil.batchEvalMissClassified(model, trainX, trainY, inputs.batchSize);
        LOG.info("Missclassified train : " + numMissClas + " of a total of " + trainY.lastDim());

        // Print out misclassified test
        model.testRunning = true;
        numMissClas = EvalUtil.batchEvalMissClassified(model, testX, testY, inputs.batchSize);
        LOG.info("Missclassified test : " + numMissClas + " of a total of " + testY.lastDim());
    }
}