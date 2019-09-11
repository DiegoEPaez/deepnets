package optim.sgd;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import optim.BatchSample;
import optim.sgd.plot.PlotLFType;
import optim.sgd.plot.PlotLossFunctionError;
import tensor.DoubleTensor;

/**
 * When using Regularization lambda penalty needs to be adjusted by batchSize /
 * trainingDataSize
 * @author diego_paez
 */
public class SGDOptim{

    public double[] optim(SGDInputs inputs) {

        BatchSample sample = new BatchSample(inputs.X, inputs.y, inputs.yWeights,
                inputs.batchSize);
        BatchSample plottingSample = null, annealSample;
        
        DoubleTensor sX, sy, st, fixedPlotX, fixedPlotY, fixedAnnealX, fixedAnnealY;
        DoubleTensor fixedPlotT, fixedAnnealT;
        fixedPlotX = null; fixedPlotY = null; fixedPlotT = null;
        fixedAnnealX = null; fixedAnnealY = null; fixedAnnealT = null;
        PlotLossFunctionError lossF = new PlotLossFunctionError(inputs.updater.stepSize);
        double err = 0, lastAnnealErr = 0;
        boolean annealErrCalc = false;
        int numBatches;
        double lastBatch = 0.0;

        numBatches = sample.getNumBatches();

        // batch for plotting when using a FIXEDBATCH
        if(inputs.plotLFOptions.type == PlotLFType.FIXEDBATCH){
            plottingSample = new BatchSample(inputs.X, inputs.y, inputs.yWeights,
                    inputs.plotLFOptions.batchPlotSize);
            plottingSample.initRandBatch();
            fixedPlotX = plottingSample.getBatchX();
            fixedPlotY = plottingSample.getBatchY();
            fixedPlotT = plottingSample.getBatchT();
        }

        if(inputs.annealEvery > 0){
            if(inputs.sameBatchPlotAndAnneal && inputs.plotLFOptions.type == PlotLFType.FIXEDBATCH){
                annealSample = plottingSample;
                fixedAnnealX = fixedPlotX;
                fixedAnnealY = fixedPlotY;
                fixedAnnealT = fixedPlotT;
            } else{
                annealSample = new BatchSample(inputs.X, inputs.y, inputs.yWeights,
                        Math.min(1000,Math.max(250,(int)(inputs.X.lastDim() * 0.05))));
                annealSample.initRandBatch();
                fixedAnnealX = annealSample.getBatchX();
                fixedAnnealY = annealSample.getBatchY();
                fixedAnnealT = annealSample.getBatchT();
            }
            
        }

        System.out.println("Step size: " + inputs.updater.stepSize);
        double propEx;
        for(int i = 0; i < inputs.epochs; i++){
            
            sample.initRandBatch();
            
            for(int j = 0; j < numBatches; j++){
                // update
                sX = sample.getBatchX();
                sy = sample.getBatchY();
                st = sample.getBatchT();
                propEx = sX.lastDim() / (double) inputs.X.lastDim();
                err = inputs.updater.update(inputs.x, inputs.fun, inputs.grad,
                        sX, sy, st, propEx);

                if(inputs.plotLFOptions.type == PlotLFType.ALLBATCHES){
                    if(inputs.plotLFOptions.startIter <= j + i * numBatches){

                        // Scale last batch to have a consistent error in case lastBatch is smaller
                        if(numBatches > 1){
                            if(j == numBatches - 2)
                                lastBatch = sX.dims[sX.dims.length - 1];
                            if(j == numBatches - 1)
                                err *= lastBatch /  (double)sX.dims[sX.dims.length - 1];
                        }

                        lossF.addData(err);

                        if(!lossF.isVisible())
                            lossF.setVisible(true);
                    }
                }

                // move to next batch
                sample.nextBatch();
            }

            // Calculations after each epoch

            // anneal
            if(inputs.annealEvery > 0){
                if(i == 0){
                    propEx = fixedAnnealX.lastDim() / (double) inputs.X.lastDim();
                    err = inputs.fun.value(inputs.x, fixedAnnealX, fixedAnnealY,
                            fixedAnnealT,propEx);
                    lastAnnealErr = err;
                    annealErrCalc = true; // used to avoid recalculating in the plots part the error

                } else if(i % inputs.annealEvery == 0){
                    if(inputs.alwaysAnneal){
                        inputs.updater.stepSize /= inputs.annealRate;
                    } else{
                        propEx = fixedAnnealX.lastDim() / (double) inputs.X.lastDim();
                        err = inputs.fun.value(inputs.x, fixedAnnealX, fixedAnnealY,
                                fixedAnnealT,propEx);

                        if(err - lastAnnealErr > 0){ // only anneal if error has gone up
                            inputs.updater.stepSize /= inputs.annealRate;
                            System.out.println("Step size: " + inputs.updater.stepSize);
                        }
                        lastAnnealErr = err;
                        annealErrCalc = true;
                    }
                }
            }

            // Plots
            if(inputs.plotLFOptions.type == PlotLFType.FIXEDBATCH){
                if(inputs.plotLFOptions.startEpoch <= i){
                    if(!annealErrCalc || !inputs.sameBatchPlotAndAnneal){
                        propEx = fixedAnnealX.lastDim() / (double) inputs.X.lastDim();
                        err = inputs.fun.value(inputs.x, fixedPlotX, fixedPlotY,
                                fixedPlotT,propEx);
                    }
                    lossF.addData(err);
                }
                if(!lossF.isVisible())
                    lossF.setVisible(true);
            }

            // Save
            if(inputs.saveWeights == true){
               storeWeights(inputs.file,inputs.x);
            }

            // Reset
            annealErrCalc = false;
        }

        return inputs.x;
    }

    public static void storeWeights(String file, double[] weights){
        DataOutputStream os = null;
        try {
            os = new DataOutputStream(new FileOutputStream(file));
            for(int i = 0; i < weights.length; i++)
                os.writeDouble(weights[i]);
        }  catch(IOException e){
            e.printStackTrace();
        } finally {
            try {
                os.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

}
