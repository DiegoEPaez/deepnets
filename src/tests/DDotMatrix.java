/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package tests;

import java.util.Random;
import org.jblas.DoubleMatrix;

/**
 *
 * @author diego_paez
 */
public class DDotMatrix {

    public static void main(String [] args){
        int num = 32000000;
        double[] mat1 = new double[num];
        double[] mat2 = new double[num];
        double[] mat3 = new double[num];
        Random rand = new Random();
        for(int i = 0; i < num; i++){
            mat1[i] = rand.nextDouble();
        }
        for(int i = 0; i < num; i++){
            mat2[i] = rand.nextDouble();
        }

        long time = System.currentTimeMillis();
        DoubleMatrix a = new DoubleMatrix(mat1);
        DoubleMatrix b = new DoubleMatrix(mat2);
        DoubleMatrix c = new DoubleMatrix(mat3);
        a.muli(b,c);
        System.out.println((System.currentTimeMillis() - time) / 1E3);

        time = System.currentTimeMillis();
        for(int i = 0; i < mat1.length; i++){
            mat3[i] = mat1[i] * mat2[i];
        }
        System.out.println((System.currentTimeMillis() -  time) / 1E3);
    }
}
