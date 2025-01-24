#include <stdio.h> // Standard I/O library for printf function
#include "hashmap.c" // Custom header file containing HashMap implementation

// Function to test the basic `put` and `get` functionality of the HashMap
void test_put_and_get() {
    printf("Running test_put_and_get...\n");

    // Create a new HashMap instance
    HashMap *map = create_hashmap();

    // Add two key-value pairs to the HashMap
    put(map, "key1", "value1");
    put(map, "key2", "value2");

    // Retrieve the values associated with the keys
    char *value1 = get(map, "key1");
    char *value2 = get(map, "key2");

    // Validate that the retrieved values match the inserted ones
    if (value1 && strcmp(value1, "value1") == 0 &&
        value2 && strcmp(value2, "value2") == 0) {
        printf("PASS: test_put_and_get\n"); // Test passes if all conditions are met
    } else {
        printf("FAIL: test_put_and_get\n"); // Test fails otherwise
    }

    // Free the memory used by the HashMap
    free_hashmap(map);
}

// Function to test overwriting an existing key in the HashMap
void test_overwrite_value() {
    printf("Running test_overwrite_value...\n");

    // Create a new HashMap instance
    HashMap *map = create_hashmap();

    // Add a key-value pair to the HashMap
    put(map, "key1", "value1");

    // Overwrite the value for the same key
    put(map, "key1", "new_value1");

    // Retrieve the updated value associated with the key
    char *value = get(map, "key1");

    // Validate that the value has been updated
    if (value && strcmp(value, "new_value1") == 0) {
        printf("PASS: test_overwrite_value\n"); // Test passes if value is correctly updated
    } else {
        printf("FAIL: test_overwrite_value\n"); // Test fails otherwise
    }

    // Free the memory used by the HashMap
    free_hashmap(map);
}

// Function to test the behavior of the HashMap when querying a nonexistent key
void test_nonexistent_key() {
    printf("Running test_nonexistent_key...\n");

    // Create a new HashMap instance
    HashMap *map = create_hashmap();

    // Attempt to retrieve a value for a key that does not exist in the HashMap
    char *value = get(map, "nonexistent");

    // Validate that the return value is NULL (indicating the key does not exist)
    if (value == NULL) {
        printf("PASS: test_nonexistent_key\n"); // Test passes if NULL is returned
    } else {
        printf("FAIL: test_nonexistent_key\n"); // Test fails otherwise
    }

    // Free the memory used by the HashMap
    free_hashmap(map);
}

// Function to test adding and retrieving a large number of elements in the HashMap
void test_large_number_of_elements() {
    printf("Running test_large_number_of_elements...\n");

    // Create a new HashMap instance
    HashMap *map = create_hashmap();
    int all_correct = 1; // Flag to track if all elements are correct

    // Insert 10 key-value pairs into the HashMap
    for (int i = 0; i < 10; i++) {
        char key[MAX_KEY_SIZE];
        char value[MAX_VALUE_SIZE];

        // Generate keys and values dynamically (e.g., "key0" -> "val0")
        snprintf(key, MAX_KEY_SIZE, "key%d", i);
        snprintf(value, MAX_VALUE_SIZE, "val%d", i);
        put(map, key, value); // Add each key-value pair to the HashMap
    }

    // Validate that all inserted key-value pairs are retrievable and correct
    for (int i = 0; i < 10; i++) {
        char key[MAX_KEY_SIZE];
        char expected[MAX_VALUE_SIZE];

        // Generate expected keys and values
        snprintf(key, MAX_KEY_SIZE, "key%d", i);
        snprintf(expected, MAX_VALUE_SIZE, "val%d", i);

        // Retrieve the value for the current key
        char *value = get(map, key);

        // Check if the value matches the expected value
        if (!value || strcmp(value, expected) != 0) {
            all_correct = 0; // Mark as incorrect if any value mismatches
            break;
        }
    }

    // Print the test result based on the correctness flag
    if (all_correct) {
        printf("PASS: test_large_number_of_elements\n"); // Test passes if all values are correct
    } else {
        printf("FAIL: test_large_number_of_elements\n"); // Test fails otherwise
    }

    // Free the memory used by the HashMap
    free_hashmap(map);
}

// Main function to run all the test cases
int main() {
    test_put_and_get();            // Test adding and retrieving elements
    test_overwrite_value();        // Test overwriting an existing key
    test_nonexistent_key();        // Test querying a nonexistent key
    test_large_number_of_elements(); // Test scalability with multiple elements

    return 0; // Return 0 to indicate successful execution
}
