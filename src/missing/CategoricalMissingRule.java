/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package missing;

/**
 *
 * @author diego_paez
 */
public interface CategoricalMissingRule {

    public boolean isPassRequired();
    public void addData(String data);
    public void advanceVarIndex();
    public void passFinished();
    public String getReplaceValue(int numVar);
}
