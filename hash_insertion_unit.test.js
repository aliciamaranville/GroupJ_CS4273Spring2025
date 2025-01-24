// Group of tests for validating the behavior of the JavaScript Map object
describe('HashMap tests', () => {
    
  /**
   * Test case: Adding a single element to the Map
   * - This verifies the basic functionality of the `set` and `get` methods.
   * - Ensures that a key-value pair can be added to the Map and retrieved correctly.
   */
  test('should add an element to the HashMap', () => {
      // Create a new Map instance
      let hashmap = new Map();

      // Add a key-value pair to the map
      hashmap.set('key1', 'value1');

      // Verify that the value for 'key1' is correctly stored and retrieved
      expect(hashmap.get('key1')).toBe('value1');
  });

  /**
   * Test case: Overwriting an existing key in the Map
   * - This checks that when a key is already present, setting a new value updates the key's value.
   * - Ensures the Map's behavior follows the expected key-value replacement logic.
   */
  test('should overwrite an existing key', () => {
      // Create a new Map instance
      let hashmap = new Map();

      // Add an initial key-value pair
      hashmap.set('key1', 'value1');

      // Overwrite the value for the same key
      hashmap.set('key1', 'newValue');

      // Verify that the value for 'key1' is updated to 'newValue'
      expect(hashmap.get('key1')).toBe('newValue');
  });

  /**
   * Test case: Handling a large number of elements in the Map
   * - This tests the Map's capacity and efficiency when storing a large dataset.
   * - Ensures the Map can handle a large number of entries without performance degradation or errors.
   */
  test('should handle large number of elements', () => {
      // Create a new Map instance
      let hashmap = new Map();

      // Add 10,000 elements to the Map
      for (let i = 0; i < 10000; i++) {
          hashmap.set(`key${i}`, `value${i}`); // Key: "key{i}", Value: "value{i}"
      }

      // Verify that the Map contains exactly 10,000 elements
      expect(hashmap.size).toBe(10000);
  });

});
