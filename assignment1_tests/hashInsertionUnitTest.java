import static org.junit.Assert.*; // Importing JUnit assertion methods to perform tests
import org.junit.Test;           // Importing the @Test annotation to designate test methods
import java.util.HashMap;       // Importing the HashMap class for creating and managing a hashmap

// Test class for validating the behavior of the HashMap class
public class HashMapTest {

    /**
     * Test case to verify adding a single element to a HashMap.
     * - This tests the basic functionality of the `put` and `get` methods.
     * - It ensures that an element added to the map is retrievable using its key.
     */
    @Test
    public void testAddElement() {
        // Initialize a new HashMap
        HashMap<String, String> hashmap = new HashMap<>();

        // Add a key-value pair to the map
        hashmap.put("key1", "value1");

        // Verify that the value associated with "key1" is correctly stored and retrieved
        assertEquals("value1", hashmap.get("key1"));
    }

    /**
     * Test case to verify the behavior when overwriting an existing key in the HashMap.
     * - This checks that adding a new value for an already existing key replaces the old value.
     * - It ensures the map's key-value replacement logic functions as expected.
     */
    @Test
    public void testOverwriteElement() {
        // Initialize a new HashMap
        HashMap<String, String> hashmap = new HashMap<>();

        // Add an initial key-value pair
        hashmap.put("key1", "value1");

        // Overwrite the value for the same key
        hashmap.put("key1", "newValue");

        // Verify that the value for "key1" is updated to "newValue"
        assertEquals("newValue", hashmap.get("key1"));
    }

    /**
     * Test case to verify the behavior of a HashMap when adding a large number of elements.
     * - This tests the map's scalability and performance with a significant number of entries.
     * - It ensures that the map handles a large dataset without losing data or crashing.
     */
    @Test
    public void testLargeNumberOfElements() {
        // Initialize a new HashMap
        HashMap<String, String> hashmap = new HashMap<>();

        // Add 10,000 elements to the map
        for (int i = 0; i < 10000; i++) {
            hashmap.put("key" + i, "value" + i); // Key format: "key{i}", Value format: "value{i}"
        }

        // Verify that the map contains exactly 10,000 elements
        assertEquals("The map should contain 10000 elements", 10000, hashmap.size());
    }
}
