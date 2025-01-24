describe('HashMap tests', () => {
    
  // Test to add a simple element
  test('should add an element to the HashMap', () => {
      let hashmap = new Map();
      hashmap.set('key1', 'value1');
      expect(hashmap.get('key1')).toBe('value1');
  });

  // Edge case: Overwriting an existing key
  test('should overwrite an existing key', () => {
      let hashmap = new Map();
      hashmap.set('key1', 'value1');
      hashmap.set('key1', 'newValue');
      expect(hashmap.get('key1')).toBe('newValue');
  });

  // Edge case: Handling a large number of elements
  test('should handle large number of elements', () => {
      let hashmap = new Map();
      for (let i = 0; i < 10000; i++) {
          hashmap.set(`key${i}`, `value${i}`);
      }
      expect(hashmap.size).toBe(10000);
  });

});
