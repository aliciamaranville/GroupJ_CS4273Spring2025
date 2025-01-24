import unittest

class TestHashMap(unittest.TestCase):
    
    # Test to add a simple element
    def test_add_element(self):
        hashmap = {}
        hashmap["key1"] = "value1"
        self.assertEqual(hashmap["key1"], "value1")

    # Edge case: Overwriting an existing key
    def test_overwrite_element(self):
        hashmap = {}
        hashmap["key1"] = "value1"
        hashmap["key1"] = "newValue"
        self.assertEqual(hashmap["key1"], "newValue")
    
    # Edge case: Handling a large number of elements
    def test_large_number_of_elements(self):
        hashmap = {}
        for i in range(10000):
            hashmap[f"key{i}"] = f"value{i}"
        self.assertEqual(len(hashmap), 10000)
    
if __name__ == "__main__":
    unittest.main()
