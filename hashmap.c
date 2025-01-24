#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HASHMAP_SIZE 10
#define MAX_KEY_SIZE 32
#define MAX_VALUE_SIZE 64

typedef struct Entry {
    char key[MAX_KEY_SIZE];
    char value[MAX_VALUE_SIZE];
    struct Entry *next;
} Entry;

typedef struct HashMap {
    Entry *buckets[HASHMAP_SIZE];
} HashMap;

unsigned int hash(const char *key) {
    unsigned int hash = 0;
    while (*key) {
        hash = (hash * 31) + *key++; // Simple hash function
    }
    return hash % HASHMAP_SIZE;
}

HashMap *create_hashmap() {
    HashMap *map = malloc(sizeof(HashMap));
    if (map) {
        for (int i = 0; i < HASHMAP_SIZE; i++) {
            map->buckets[i] = NULL;
        }
    }
    return map;
}

void put(HashMap *map, const char *key, const char *value) {
    unsigned int index = hash(key);
    Entry *entry = map->buckets[index];

    // Check if key already exists
    while (entry) {
        if (strncmp(entry->key, key, MAX_KEY_SIZE) == 0) {
            strncpy(entry->value, value, MAX_VALUE_SIZE);
            return;
        }
        entry = entry->next;
    }

    // Add a new entry
    Entry *new_entry = malloc(sizeof(Entry));
    if (new_entry) {
        strncpy(new_entry->key, key, MAX_KEY_SIZE);
        strncpy(new_entry->value, value, MAX_VALUE_SIZE);
        new_entry->next = map->buckets[index];
        map->buckets[index] = new_entry;
    }
}

char *get(HashMap *map, const char *key) {
    unsigned int index = hash(key);
    Entry *entry = map->buckets[index];

    while (entry) {
        if (strncmp(entry->key, key, MAX_KEY_SIZE) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

void free_hashmap(HashMap *map) {
    for (int i = 0; i < HASHMAP_SIZE; i++) {
        Entry *entry = map->buckets[i];
        while (entry) {
            Entry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(map);
}
