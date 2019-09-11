package tensor;

/**
 *
 * @author diego_paez
 */
public class TensorFunctions {

    // constant such that 1 + exp(-limExp) = 1.0 in double arithmetic is 36.0436533891172, 36 is a little below
    public static final double smallConst = Math.exp(-36);

    public static DoubleTensor expi(DoubleTensor x){
        for(int i = 0; i < x.data.size();i++){
            x.data.setQuick(i, Math.exp(x.data.getQuick(i)));
        }

        return x;
    }

    public static DoubleTensor powi(DoubleTensor x, double power){
        for(int i = 0; i < x.data.size();i++){
            x.data.setQuick(i, Math.pow(x.data.getQuick(i),power));
        }

        return x;
    }

    public static DoubleTensor logi(DoubleTensor x){
        for(int i = 0; i < x.data.size();i++){
            x.data.setQuick(i, Math.log(x.data.getQuick(i)));
        }

        return x;
    }

    public static DoubleTensor sqrti(DoubleTensor x){
        for(int i = 0; i < x.data.size();i++){
            x.data.setQuick(i, Math.sqrt(x.data.getQuick(i)));
        }

        return x;
    }
}
