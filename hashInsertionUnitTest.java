import static org.junit.Assert.*;
import org.junit.Test;
import java.util.HashMap;

public class HashMapTest {

    // Test to add a simple element
    @Test
    public void testAddElement() {
        HashMap<String, String> hashmap = new HashMap<>();
        hashmap.put("key1", "value1");
        assertEquals("value1", hashmap.get("key1"));
    }

    // Edge case: Overwriting an existing key
    @Test
    public void testOverwriteElement() {
        HashMap<String, String> hashmap = new HashMap<>();
        hashmap.put("key1", "value1");
        hashmap.put("key1", "newValue");
        assertEquals("newValue", hashmap.get("key1"));
    }

    // Edge case: Adding a large number of elements (for performance)
    @Test
    public void testLargeNumberOfElements() {
        HashMap<String, String> hashmap = new HashMap<>();
        for (int i = 0; i < 10000; i++) {
            hashmap.put("key" + i, "value" + i);
        }
        assertEquals("The map should contain 10000 elements", 10000, hashmap.size());
    }
}
