import unittest

class TestHashMap(unittest.TestCase):
    def test_add_element(self):
        hashmap = {}
        hashmap['key1'] = 'value1'
        self.assertEqual(hashmap['key1'], 'value1')

if __name__ == '__main__':
    unittest.main()
