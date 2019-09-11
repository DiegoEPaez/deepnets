package wregul;

import core.NeuralNetModel;

/**
 *
 * @author diego_paez
 */
public abstract class WeightRegularization {

    public double lambda;

    public WeightRegularization(double lambda) {
        this.lambda = lambda;
    }

    public abstract double eval(NeuralNetModel model, double lambdaAdjus);
    public abstract void updateWeights(NeuralNetModel model, double lambdaAdjus);
}
