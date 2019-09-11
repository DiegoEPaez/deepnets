/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package missing;

/**
 *
 * @author diego_paez
 */
public interface NumericMissingRule{

    public boolean isPassRequired();
    public void addData(double data);
    public void advanceVarIndex();
    public void passFinished();
    public double getReplaceValue(int numVar);
}
