import static org.junit.Assert.*;
import org.junit.Test;
import java.util.HashMap;

public class HashMapTest {
    @Test
    public void testAddElement() {
        HashMap<String, String> hashmap = new HashMap<>();
        hashmap.put("key1", "value1");
        assertEquals("value1", hashmap.get("key1"));
    }
}
