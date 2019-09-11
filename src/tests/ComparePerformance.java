/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package tests;

import org.apache.commons.math3.special.Gamma;

/**
 *
 * @author diego_paez
 */
public class ComparePerformance {

    public static void main(String [] args){

        long time,time2;
        long t1, t2;
        double res;
        for(int i = 0; i < 1000; i++){
            time = System.nanoTime();
            res = Gamma.logGamma(i + 1);
            time2 = System.nanoTime();
            t1 = (time2 - time);
            System.out.println("Time 1: " + t1);


            time = System.nanoTime();
            res = 0;
            for(int j = 1; j <= i; j++){
                res += Math.log(j);
            }
            time2 = System.nanoTime();
            t2 = (time2 - time);
            System.out.println("Time 2: " + t2);

            if(t2 > t1)
                System.out.println("Flip at " + i);



        }
    }

}
