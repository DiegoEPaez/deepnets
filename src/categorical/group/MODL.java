package categorical.group;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import org.apache.log4j.Logger;

/**
 * Implementation of the algorithm MODL to group classes. The algorithm is
 * described in the article:
 * "A Bayes Optimal Approach for Partitioning the Values of Categorical Attributes".
 * by Marc Boullé
 * @author diego_paez
 */
public class MODL {

    private static final Logger LOG = Logger.getLogger(MODL.class);

    /**
     * Greedy search to find an optimal grouping using MODL evaluation.
     * @param values
     * @param classes 
     */
    public void optimalGrouping(ArrayList<String> values, ArrayList<String> classes){
        EvalMODL ev = new EvalMODL();
        LinkedHashSet<String> cSet = new LinkedHashSet<>();
        LinkedHashSet<String> vSet = new LinkedHashSet<>();
        for(int i = 0; i < classes.size(); i++){
            cSet.add(classes.get(i));
            vSet.add(values.get(i));
        }

        LinkedHashMap<String, Integer> classAss = new LinkedHashMap<>();
        int k = 0;
        for(String s: cSet){
            classAss.put(s, k);
            k++;
        }

        LinkedHashMap<String, Integer> currGroupAss = new LinkedHashMap<>();
        k = 0;
        for(String s: vSet){
            currGroupAss.put(s, k);
            k++;
        }

        ArrayList<Integer> classInt = new ArrayList<>();
        ArrayList<Integer> groupInt = new ArrayList<>();

        for(int i = 0; i < classes.size(); i++){
            classInt.add(classAss.get(classes.get(i)));
            groupInt.add(currGroupAss.get(values.get(i)));
        }
        int numGroups = vSet.size();

        double baseValue = ev.evalGroup(vSet.size(), values, cSet.size(), classInt,
                numGroups, groupInt);

        // merge groups to look for the minimum evaluation, using a greedy approach
        LinkedHashMap<String, Integer> prevGroupAss;
        int val;
        double[] mergeVals;
        int[] m1S, m2S;
        int nmer, pos;
        double min;
        while(numGroups > 1){
            numGroups--;
            prevGroupAss = currGroupAss;

            // merge
            currGroupAss = new LinkedHashMap<>();
            nmer = numGroups;
            mergeVals = new double[(nmer * (nmer + 1)) / 2];
            m1S = new int[(nmer * (nmer + 1)) / 2];
            m2S = new int[(nmer * (nmer + 1)) / 2];
            nmer = 0;
            for(int m1 = 0; m1 < numGroups; m1++){
                for(int m2 = m1 + 1; m2 < numGroups + 1; m2++){
                    currGroupAss.clear();
                    for(String s:prevGroupAss.keySet()){
                        val = prevGroupAss.get(s);
                        if(val < m2){
                            currGroupAss.put(s, val);
                        } else if(val == m2){
                            currGroupAss.put(s, m1);
                        } else{
                            currGroupAss.put(s, val - 1);
                        }
                    }
                    groupInt.clear();
                    for(int i = 0; i < classes.size(); i++){
                        groupInt.add(currGroupAss.get(values.get(i)));
                    }

                    mergeVals[nmer] = ev.evalGroup(vSet.size(), values, cSet.size(), classInt,
                        numGroups, groupInt);
                    m1S[nmer] = m1;
                    m2S[nmer] = m2;
                    nmer++;
                    if(nmer % 100 == 0)
                        LOG.info("nmer " + nmer);
                }
            }

            // find min
            pos = -1;
            min = baseValue;
            for(int i = 0; i < mergeVals.length; i++){
                if(min > mergeVals[i]){
                    pos = i;
                    min = mergeVals[i]; 
                }
            }
            if(pos == -1){
                currGroupAss = prevGroupAss;
                break;
            }

            currGroupAss.clear();
            for(String s:prevGroupAss.keySet()){
                val = prevGroupAss.get(s);
                if(val < m2S[pos]){
                    currGroupAss.put(s, val);
                } else if(val == m2S[pos]){
                    currGroupAss.put(s, m1S[pos]);
                } else{
                    currGroupAss.put(s, val - 1);
                }
            }
            baseValue = min;
            
        }

        LOG.info(baseValue);
        LOG.info(currGroupAss);
    }
}