/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package tests;

import categorical.group.MODL;
import io.ReadCSV;
import java.io.File;
import java.util.ArrayList;

/**
 *
 * @author diego_paez
 */
public class TestMODL {

    public static void main(String[] args) {
        ArrayList<String> values = new ArrayList<>();
        ArrayList<String> classes = new ArrayList<>();
        test2(values, classes);

        MODL m = new MODL();

        m.optimalGrouping(values, classes);
    }

    private static void test1(ArrayList<String> values, ArrayList<String> classes){
        values.add("A");
        values.add("A");
        values.add("C");
        values.add("B");
        values.add("A");
        values.add("A");
        values.add("B");
        values.add("B");
        values.add("C");
        values.add("B");
        values.add("C");

        classes.add("SAR");
        classes.add("NO_SAR");
        classes.add("SAR");
        classes.add("NO_SAR");
        classes.add("SAR");
        classes.add("SAR");
        classes.add("NO_SAR");
        classes.add("NO_SAR");
        classes.add("SAR");
        classes.add("SAR");
        classes.add("NO_SAR");
    }

    private static void test2(ArrayList<String> values, ArrayList<String> classes){
        File fr = new File("C:\\Users\\DP12577\\Documents\\Analytics\\Pruebas_Variables\\"
                + "20161010_Actividad_CP_Sucursal\\ActividadHist2.csv");
        ReadCSV rc = new ReadCSV(fr);
        // read num. instances
        rc.readHeaders();

        while(rc.next()){
            values.add(rc.get(0));
            classes.add(rc.get(0));
        }
    }
}
