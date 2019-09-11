package optim.sgd.plot;

/**
 *
 * @author diego_paez
 */
public class PlotLFOptions {

    public PlotLFType type;
    public int startEpoch;
    public int startIter;
    public int batchPlotSize;

    public PlotLFOptions(PlotLFType type, int startEpoch, int startIter, int batchPlotSize) {
        this.type = type;
        this.startEpoch = startEpoch;
        this.startIter = startIter;
        this.batchPlotSize = batchPlotSize;
    }    
}
