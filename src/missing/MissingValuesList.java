package missing;

import java.util.HashSet;

/**
 *
 * @author diego_paez
 */
public class MissingValuesList {

    private static HashSet<String> missingValues = new HashSet();

    public static void addValues(String str){
        missingValues.add(str);
    }

    public static boolean isMissingValue(String str){
        return missingValues.contains(str);
    }

}
