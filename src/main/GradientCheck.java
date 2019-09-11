package main;

import core.NeuralNetFunction;
import core.NeuralNetGradient;
import core.NeuralNetModel;
import init.He;
import init.Xavier;
import io.ReadMNIST;
import java.util.Random;
import layer.activation.ActivationLayer;
import layer.activation.ReLU;
import layer.activation.SoftMaxLayer;
import layer.subsampling.MeanPooling2DLayer;
import layer.weight.Convolution2DLayer;
import layer.weight.InnerProductLayer;
import loss.CrossEntropy;
import optim.BatchSample;
import tensor.DoubleTensor;

/**
 *
 * @author diego_paez
 */
public class GradientCheck {

    public static void main(String [] args){
        System.out.println("Le Net Gradient test: ");
        System.out.println("Avg. Error: " + testLeNet());

        /*System.out.println("Gradient test: ");
        System.out.println("Avg. Error: " + test2());*/

    }

    public static double testLeNet(){
                System.out.println("Running Neural Net Regression");
        System.out.println("Loading MNIST Data");
        DoubleTensor[] trainTest = ReadMNIST.loadMNIST(false,100,0);

        DoubleTensor trainX = trainTest[0];
        DoubleTensor trainY = trainTest[1];
        DoubleTensor testX = trainTest[2];
        DoubleTensor testY = trainTest[3];

        NeuralNetModel model = new NeuralNetModel(new CrossEntropy());

        model.addLayer(new Convolution2DLayer(new He(), 20, 5, 5, 1, 1,Convolution2DLayer.ConvolveMethod.FFT))
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new MeanPooling2DLayer(2, 2))
             .addLayer(new InnerProductLayer(new He(), 10))
             .addLayer(new SoftMaxLayer());
        
        /*model.addLayer(new Convolution2DLayer(new He(), 6, 5, 5))  // reduced to 28 x 28 use 5 x 5 conv & 6 filters
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new MeanPooling2DLayer(2, 2))               // reduced to 14 x 14
             .addLayer(new Convolution2DLayer(new He(), 16, 5, 5)) // reduced to 10 x 10
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new MeanPooling2DLayer(2, 2))               // reduced to 5 x 5
             .addLayer(new Convolution2DLayer(new He(), 128, 5, 5)) // reduced to 1 x 1
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new InnerProductLayer(new He(), 84)) // use 84 neurons on the 128 inputs
             .addLayer(new ActivationLayer(new ReLU()))
             .addLayer(new InnerProductLayer(new He(), 10)) // use 10 neurons on the 84 inputs
             .addLayer(new SoftMaxLayer());*/

        model.setTrainInputs(trainX, trainY, null);
        model.initWeights();

        NeuralNetFunction fun = new NeuralNetFunction(model);
        NeuralNetGradient grad = new NeuralNetGradient(model);


        BatchSample sample = new BatchSample(trainX, trainY, null,
                250);
        sample.initRandBatch();
        DoubleTensor xSample = sample.getBatchX();
        DoubleTensor ySample = sample.getBatchY();

        fun.value(model.theta,xSample,ySample,null, 250.0 / trainX.lastDim());
        double[] theoGrad = grad.value(model.theta,xSample,ySample,null,250.0 / trainX.lastDim());

        Random rand = new Random();
        int iter = 50;
        double error = 0;
        double numDer = 0, theoDer;
        int next;
        next = 0;
        for (int k = 0; k < iter; k++) {
            next = rand.nextInt(model.theta.length);
            numDer = centeredIthGradVal2(model.theta, fun, next,xSample,ySample,
                    null,250.0 / trainX.lastDim());//350
            theoDer = theoGrad[next];
            error += Math.abs(numDer
                    - theoDer);
            System.out.println("numDer: " + numDer + " theoDer:" + theoDer);
        }
        error /= iter;

        return error;
    }

    public static double test2(){
        double[] X = new double[]{
            0,1,1,0,
            0,0,1,1,
            1,0,0,0,
            0,0,0,1
        };
        double[] y = new double[]{0,0,1,1};

        DoubleTensor trainX = new DoubleTensor(X,new int[]{4,4});
        DoubleTensor trainY = new DoubleTensor(y,new int[]{4});

        NeuralNetModel model = new NeuralNetModel(new CrossEntropy());
        
        model.addLayer(new InnerProductLayer(new Xavier(123), 2))
             .addLayer(new SoftMaxLayer());

        model.setTrainInputs(trainX, trainY, null);
        model.initWeights(); // model.loadWeights(file)...

        NeuralNetFunction fun = new NeuralNetFunction(model);
        NeuralNetGradient grad = new NeuralNetGradient(model);

        fun.value(model.theta,trainX,trainY,null, 250.0 / trainX.lastDim());
        double[] theoGrad = grad.value(model.theta,trainX,trainY,null,250.0 / trainX.lastDim());

        Random rand = new Random(123);
        int iter = 50;
        double error = 0;
        double numDer = 0, theoDer;
        int next;
        next = 0;
        for (int k = 0; k < iter; k++) {
            next = rand.nextInt(model.theta.length);
            numDer = centeredIthGradVal2(model.theta, fun, next,trainX,trainY,
                    null,250.0 / trainX.lastDim());//350
            theoDer = theoGrad[next];
            error += Math.abs(numDer
                    - theoDer);
            System.out.println("numDer: " + numDer + " theoDer:" + theoDer);
        }
        error /= iter;

        return error;

    }

    public static double centeredIthGradVal2(double[] theta,
            NeuralNetFunction function, int i, DoubleTensor xBatch,
            DoubleTensor yBatch, DoubleTensor tBatch, double lambda) {
        double h = 1e-7;
        double aux;

        theta[i] = theta[i] + h;
        aux = function.value(theta, xBatch, yBatch, tBatch, lambda);

        theta[i] = theta[i] - h - h;
        aux -= function.value(theta, xBatch, yBatch, tBatch, lambda);
        theta[i] = theta[i] + h;

        return aux / (2.0 * h);
    }
}
