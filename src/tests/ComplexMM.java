/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package tests;

import org.jblas.ComplexDouble;
import org.jblas.NativeBlas;

/**
 *
 * @author diego_paez
 */
public class ComplexMM {

    public static void main(String [] args){
        test2();
    }

    private static void test1(){
        double[] aMat = new double[]{
            1.0,2.0,
            3.0,4.0,
            5.0,6.0,
            7.0,8.0};
        double[] bMat = new double[]{
            2.0,1.0,
            4.0,3.0,
            5.0,6.0,
            7.0,8.0};
        double[] cMat;

        // As specified in LAPACK documentation for zgemm, m = rows of C,
        // n = cols of C, k = mult dim
        int  m, n, k;
        char aT, bT;

        m = 2;
        k = 2;
        n = 2;

        cMat = new double[m * n * 2];

        NativeBlas.zgemm('N', 'N', m, n, k, new ComplexDouble(1.0), aMat,
                0, m, bMat, 0, n, new ComplexDouble(0.0), cMat, 0, m);

        System.out.println("d");
    }

    private static void test2(){
        double[] aMat = new double[]{
            1.0,2.0,
            3.0,4.0,
            5.0,6.0,
            7.0,8.0};
        double[] bMat = new double[]{
            2.0,1.0,
            4.0,3.0,
            5.0,6.0,
            7.0,8.0};
        double[] cMat;

        // As specified in LAPACK documentation for zgemm, m = rows of C,
        // n = cols of C, k = mult dim
        int  m, n, k;
        char aT, bT;

        m = 1;
        k = 2;
        n = 1;

        cMat = new double[m * n * 2];

        NativeBlas.zgemm('N', 'N', m, n, k, new ComplexDouble(1.0), aMat,
                2, m, bMat, 2, k, new ComplexDouble(0.0), cMat, 0, m);

        System.out.println("d");
    }
}