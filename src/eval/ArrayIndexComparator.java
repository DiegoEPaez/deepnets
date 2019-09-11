package eval;

import java.util.Comparator;

/**
 * Comparator which compares the value of 2 elements of an array as specified
 * by a pair of indices given, and returns a negative integer, zero, or a
 * positive integer if the value of the first element in the array
 * less than, equal to, or greater than the second element in the array.
 * 
 * Took the code from:
 * https://stackoverflow.com/questions/4859261/get-the-indices-of-an-array-after-sorting
 * 
 * @param <T> The type of objects the array contains.
 */
public class ArrayIndexComparator<T extends Comparable<T>> implements Comparator<Integer> {

    /**
     * Array to sort.
     */
    private final T[] array;

    /**
     * Constructor for the array to sort.
     * @param array Array to sort.
     */
    public ArrayIndexComparator(T[] array) {
        this.array = array;
    }

    /**
     * Create an array with integers from 0 to array's length -1. These integers
     * represent the indices of the array to sort
     * @return Array with integers from 0 to n - 1.
     */
    public Integer[] createIndexArray() {
        Integer[] indexes = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indexes[i] = i; // Autoboxing
        }
        return indexes;
    }

    /**
     * Given to integers which represent indices in the array to sort, return
     * negative, zero or positive if the first element in the array (given 
     * by the position of the first index) is smaller, equal or bigger.
     * @param index1 First index to compare.
     * @param index2 Second index to compare.
     * @return Theindex of the biggest between both values in the array.
     */
    @Override
    public int compare(Integer index1, Integer index2) {
        // Autounbox from Integer to int to use as array indexes
        return array[index2].compareTo(array[index1]);
    }
}
