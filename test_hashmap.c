#include <stdio.h>
#include "hashmap.c"

void test_put_and_get() {
    printf("Running test_put_and_get...\n");

    HashMap *map = create_hashmap();
    put(map, "key1", "value1");
    put(map, "key2", "value2");

    char *value1 = get(map, "key1");
    char *value2 = get(map, "key2");

    if (value1 && strcmp(value1, "value1") == 0 &&
        value2 && strcmp(value2, "value2") == 0) {
        printf("PASS: test_put_and_get\n");
    } else {
        printf("FAIL: test_put_and_get\n");
    }

    free_hashmap(map);
}

void test_overwrite_value() {
    printf("Running test_overwrite_value...\n");

    HashMap *map = create_hashmap();
    put(map, "key1", "value1");
    put(map, "key1", "new_value1");

    char *value = get(map, "key1");

    if (value && strcmp(value, "new_value1") == 0) {
        printf("PASS: test_overwrite_value\n");
    } else {
        printf("FAIL: test_overwrite_value\n");
    }

    free_hashmap(map);
}

void test_nonexistent_key() {
    printf("Running test_nonexistent_key...\n");

    HashMap *map = create_hashmap();
    char *value = get(map, "nonexistent");

    if (value == NULL) {
        printf("PASS: test_nonexistent_key\n");
    } else {
        printf("FAIL: test_nonexistent_key\n");
    }

    free_hashmap(map);
}

void test_large_number_of_elements() {
    printf("Running test_large_number_of_elements...\n");

    HashMap *map = create_hashmap();
    int all_correct = 1;

    for (int i = 0; i < 10; i++) {
        char key[MAX_KEY_SIZE];
        char value[MAX_VALUE_SIZE];
        snprintf(key, MAX_KEY_SIZE, "key%d", i);
        snprintf(value, MAX_VALUE_SIZE, "val%d", i);
        put(map, key, value);
    }

    for (int i = 0; i < 10; i++) {
        char key[MAX_KEY_SIZE];
        char expected[MAX_VALUE_SIZE];
        snprintf(key, MAX_KEY_SIZE, "key%d", i);
        snprintf(expected, MAX_VALUE_SIZE, "val%d", i);

        char *value = get(map, key);
        if (!value || strcmp(value, expected) != 0) {
            all_correct = 0;
            break;
        }
    }

    if (all_correct) {
        printf("PASS: test_large_number_of_elements\n");
    } else {
        printf("FAIL: test_large_number_of_elements\n");
    }

    free_hashmap(map);
}

int main() {
    test_put_and_get();
    test_overwrite_value();
    test_nonexistent_key();
    test_large_number_of_elements();

    return 0;
}
