package optim.sgd;

import optim.BatchFunction;
import optim.BatchGradient;
import optim.sgd.plot.PlotLFOptions;
import optim.sgd.plot.PlotLFType;
import optim.sgd.update.SGDUpdate;
import tensor.DoubleTensor;

/**
 *
 * @author diego_paez
 */
public class SGDInputs {
    public double[] x;
    public BatchFunction fun;
    public BatchGradient grad;
    public DoubleTensor X;
    public DoubleTensor y;

    // Weights for example in X
    public DoubleTensor yWeights;
    public int epochs;
    public int batchSize;
    public SGDUpdate updater;
    public int annealEvery;
    public double annealRate;
    public boolean alwaysAnneal;
    public boolean sameBatchPlotAndAnneal;
    public PlotLFOptions plotLFOptions;

    public boolean saveWeights;
    public String file;
    
    public SGDInputs(double[] x, BatchFunction fun, BatchGradient grad,
            DoubleTensor X, DoubleTensor y) {
        this.x = x;
        this.fun = fun;
        this.grad = grad;
        this.X = X;
        this.y = y;
        this.epochs = 5;
        this.annealEvery = 5; // every 5 epochs check if annealing should be done
        this.annealRate = 1.02;
        this.alwaysAnneal = false;
        this.batchSize = Math.min(X.lastDim(), 256);
        this.sameBatchPlotAndAnneal = true;
        this.plotLFOptions = new PlotLFOptions(PlotLFType.NONE,0,0,0);
        this.saveWeights = false;
        this.file = "weights.dat";
    }
}