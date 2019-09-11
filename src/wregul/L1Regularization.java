package wregul;

import layer.Layer;
import layer.weight.WeightLayer;
import core.NeuralNetModel;

/**
 * Lasso Reg. - p = 1
 * Ridge Regularization - p = 2
 * @author diego_paez
 */
public class L1Regularization extends WeightRegularization {

    public L1Regularization(double lambda) {
        super(lambda);
    }

    @Override
    public double eval(NeuralNetModel model, double lambdaAdjus) {
        WeightLayer layer;
        double penalty = 0.0;

        for(Layer l: model.layers){
            if(l instanceof WeightLayer){
                layer = (WeightLayer) l;
                for(double w: layer.weights.data.getData()){
                    penalty += Math.abs(w);
                }
            }
        }
        penalty *= (lambda * lambdaAdjus);
        return penalty;
    }

    @Override
    public void updateWeights(NeuralNetModel model, double lambdaAdjus) {
        WeightLayer layer;
        double w;
        for(Layer l: model.layers){
            if(l instanceof WeightLayer){
                layer = (WeightLayer) l;
                for(int i = 0; i < layer.weightsGrad.size(); i++){
                    w = layer.weights.getQuick(i);
                    layer.weightsGrad.setQuick(i, layer.weightsGrad.getQuick(i)
                            + lambda * lambdaAdjus * Math.signum(w));
                }
            }
        }
    }
}
