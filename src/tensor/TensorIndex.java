package tensor;

/**
 *
 * @author diego_paez
 */
public class TensorIndex {

    private int[] indices;
    private int start;
    private int end;
    private int pointer;

    public TensorIndex(int[] indices) {
        this.indices = indices;
        pointer = 0;
    }

    public TensorIndex(int start, int end) {
        this.start = start;
        this.end = end;
        pointer = start;
    }

    public int size(){
        if(indices != null){
            return indices.length;
        } else{
            return end - start;
        }
    }

    public int next(){
        if(indices != null){
            return indices[pointer++];
        } else{
            return pointer++;
        }
    }

    public void resetPointer(){
        if(indices != null){
            pointer = 0;
        } else{
            pointer = start;
        }
    }
}