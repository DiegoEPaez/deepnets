package missing;

import org.apache.commons.math3.stat.descriptive.rank.PSquarePercentile;

/**
 * Fills missing data with quantile specified (typically the median = 0.5)
 * uses PSquare Algorithm, to estimate the quantile as explained in:
 *
 * http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf
 * @author diego_paez
 */
public class FillWithPSqQuant implements NumericMissingRule{

    private PSquarePercentile[] psq;
    private double[] quantiles;
    private int currVar;

    public FillWithPSqQuant(int numVars, double p) {
        psq = new PSquarePercentile[numVars];

        for(int i = 0 ; i < numVars; i++){
            psq[i] = new PSquarePercentile(p);
        }
        quantiles = new double[numVars];
    }

    @Override
    public boolean isPassRequired() {
        return true;
    }

    @Override
    public void addData(double data) {
        if(currVar == psq.length)
            currVar = 0;

        psq[currVar].increment(data);
        currVar++;
    }

    @Override
    public double getReplaceValue(int numVar) {
        return quantiles[numVar];
    }

    @Override
    public void advanceVarIndex() {
        currVar = (currVar + 1) % psq.length;
    }

    @Override
    public void passFinished() {
        for(int i = 0 ; i < psq.length; i++){
            quantiles[i] = psq[i].getResult();
        }
    }
}
