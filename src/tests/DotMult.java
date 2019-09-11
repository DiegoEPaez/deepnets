/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package tests;

import java.util.Random;

/**
 *
 * @author diego_paez
 */
public class DotMult {
    public static void main(String [] args){
        Random rand = new Random(123);
        int size = 10000000;
        double[] data1 = new double[size];
        double[] data2 = new double[size];
        double[] result = new double[size];


        for(int i = 0; i < data1.length; i++){
            data1[i] = rand.nextDouble();
            data2[i] = rand.nextDouble();
        }

        long time = System.currentTimeMillis();
        mmult1(data1,data2,result);
        time = System.currentTimeMillis() - time;
        System.out.println("mmult1: " + time / 1E3);

        time = System.currentTimeMillis();
        mmult3(data1,data2,result);
        time = System.currentTimeMillis() - time;
        System.out.println("mmult3: " + time / 1E3);

    }

    private static void mmult1(double[] data1, double[] data2, double[] result){
        int loopSize = data1.length >> 1;
        double reI, imI, reK, imK;
        double k1, k2, k3;
        for(int i = 0; i < loopSize;){
            reI = data1[i];
            reK = data2[i];
            i++;
            imI = data1[i];
            imK = data2[i];
            i++;
            k1 = reI * (reK + imK);
            k2 = imK * (reI + imI);
            k3 = reK * (imI - reI);

            result[i - 1] += k1 - k2;
            result[i] += k1 + k3;
        }
    }

    private static void mmult2(double[] data1, double[] data2, double[] result){
        System.arraycopy(data1, 0, result, 0, data1.length);
        //info.yeppp.Core.Multiply_IV64fV64f_IV64f(result, 0, data2, 0, data1.length);
    }

    private static void mmult3(double[] data1, double[] data2, double[] result){
        int loopSize = data1.length >> 1;
        double reI, imI, reK, imK;
        double k1, k2, k3;
        for(int i = 0; i < loopSize; i++){
            data1[i] *= data2[i];
        }
    }
}
