test('adds element to hash map', () => {
    const hashmap = {};
    hashmap['key1'] = 'value1';
    expect(hashmap['key1']).toBe('value1');
  });
  