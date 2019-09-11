package types;

/**
 *
 * @author DP12577
 */
public class DataTypeProperties {

    private DataTypeM type;
    private long length;

    public DataTypeProperties(DataTypeM type, long length) {
        this.type = type;
        this.length = length;
    }

    public long getLength() {
        return length;
    }

    public void setLength(long length) {
        this.length = length;
    }

    public DataTypeM getType() {
        return type;
    }

    public void setType(DataTypeM type) {
        this.type = type;
    }
}
