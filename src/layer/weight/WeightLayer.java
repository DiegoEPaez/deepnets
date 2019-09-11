package layer.weight;

import init.Initialization;
import layer.Layer;
import tensor.DoubleTensor;

/**
 *
 * @author diego_paez
 */
public abstract class WeightLayer extends Layer{
    // weights and their gradient
    public DoubleTensor bias;
    public DoubleTensor weights;
    public DoubleTensor biasGrad;
    public DoubleTensor weightsGrad;

    // Address for weights when packed as a whole
    protected int initAddr;
    
    // Initialization scheme
    protected Initialization init;

    public abstract void initParams();
    public abstract int getNumberOfParams();
    public abstract void updateLayerWGrad(DoubleTensor chainGrad);

}
