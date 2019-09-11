package tensor;

import gnu.trove.list.array.TDoubleArrayList;

/**
 *
 * @author diego_paez
 */
public class DoubleArrayList extends TDoubleArrayList{

    public DoubleArrayList(double[] data){
        _data = data;
        _pos = data.length;
    }

    public DoubleArrayList(int length){
        super(length);
    }

    public DoubleArrayList(DoubleArrayList initData){
        super(initData);
    }

    public double[] getData(){
        return _data;
    }

    public int getPos(){
        return _pos;
    }

    public void setData(double[] data){
        _data = data;
    }

    public void setPos(int pos){
        _pos = pos;
    }

    public void addQuick( int offset, double val ) {
        _data[ offset ] += val;
    }

    public void subQuick( int offset, double val ) {
        _data[ offset ] -= val;
    }

    public void mulQuick( int offset, double val ) {
        _data[ offset ] *= val;
    }

    public void divQuick( int offset, double val ) {
        _data[ offset ] /= val;
    }
}
