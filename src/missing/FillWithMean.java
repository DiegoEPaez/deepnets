package missing;

import java.util.Arrays;

/**
 *
 * @author diego_paez
 */
public class FillWithMean implements NumericMissingRule{

    private double[] means;
    private int[] count;
    private int currVar;

    public FillWithMean(int numVars) {
        currVar = 0;

        means = new double[numVars];

        Arrays.fill(means, 0.0);
    }

    @Override
    public boolean isPassRequired() {
        return true;
    }

    @Override
    public void addData(double data) {
        if(currVar == means.length)
            currVar = 0;

        means[currVar] = data;
        count[currVar]++;
        currVar++;
    }

    @Override
    public double getReplaceValue(int numVar) {
        return means[numVar];
    }

    @Override
    public void advanceVarIndex() {
        currVar = (currVar + 1) % means.length;
    }

    @Override
    public void passFinished() {
        for(int i = 0 ; i < means.length; i++){
            if(count[i] != 0)
                means[i] /= count[i];
        }
    }

}
