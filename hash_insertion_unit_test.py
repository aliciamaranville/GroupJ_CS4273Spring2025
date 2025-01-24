import unittest  # Importing the unittest module for creating and running test cases

# Test class for validating the behavior of Python's dictionary as a HashMap
class TestHashMap(unittest.TestCase):
    
    """
    Test case to verify adding a single element to the HashMap.
    - This tests the basic functionality of adding a key-value pair.
    - It ensures that a value can be added to the dictionary and retrieved correctly using its key.
    """
    def test_add_element(self):
        # Initialize an empty dictionary (used as a HashMap)
        hashmap = {}
        
        # Add a key-value pair to the dictionary
        hashmap["key1"] = "value1"
        
        # Verify that the value associated with "key1" is correctly stored and retrieved
        self.assertEqual(hashmap["key1"], "value1")

    """
    Test case to verify the behavior when overwriting an existing key in the HashMap.
    - This tests that assigning a new value to an existing key replaces the old value.
    - Ensures the dictionary's key-value replacement logic functions as expected.
    """
    def test_overwrite_element(self):
        # Initialize an empty dictionary
        hashmap = {}
        
        # Add an initial key-value pair
        hashmap["key1"] = "value1"
        
        # Overwrite the value for the same key
        hashmap["key1"] = "newValue"
        
        # Verify that the value for "key1" is updated to "newValue"
        self.assertEqual(hashmap["key1"], "newValue")
    
    """
    Test case to verify the behavior of the HashMap when handling a large number of elements.
    - This tests the scalability of the dictionary with a significant number of entries.
    - Ensures that the dictionary can store and retrieve a large dataset without any errors.
    """
    def test_large_number_of_elements(self):
        # Initialize an empty dictionary
        hashmap = {}
        
        # Add 10,000 key-value pairs to the dictionary
        for i in range(10000):
            hashmap[f"key{i}"] = f"value{i}"  # Key format: "key{i}", Value format: "value{i}"
        
        # Verify that the dictionary contains exactly 10,000 elements
        self.assertEqual(len(hashmap), 10000)

# The following block ensures the tests run when the script is executed directly
if __name__ == "__main__":
    unittest.main()
