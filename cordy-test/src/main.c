#include <stdlib.h>
#include <string.h>

#include "../../cordy/cordy.h"

CORDY_EXTERN(get_42) {
    return INT(42);
}

CORDY_EXTERN(add2) {
    ASSERT(n == 2);
    AS_INT(int64_t left, args[0]);
    int64_t right;
    AS_INT(right, args[1]);
    return INT(left + right);
}

CORDY_EXTERN(is_nil) {
    ASSERT(n == 1);
    return BOOL(IS_NIL(args[0]));
}

CORDY_EXTERN(substring) {
    ASSERT(n == 3);
    AS_STR(str_t string, args[0]);
    AS_INT(int64_t start, args[1]);
    AS_INT(int64_t len, args[2]);

    // Allocate a buffer for the new string, in C land
    char* new = malloc(len + 1);

    // Copy the substring from the source (which is a const char*) to our new buffer, including null terminator
    memcpy(new, string + start, len);
    new[len] = '\0';

    // Copy the buffer into a Cordy string
    cordy_value_t ret = STR(new);

    // Free both the input argument, and the buffer
    // Return the new Cordy string to the caller which will take ownership of the memory
    FREE(args[0]);
    free(new);

    return ret;
}

CORDY_EXTERN(partial_sums) {
    ASSERT(n == 1);
    AS_VEC(cordy_vec_t vec, args[0]);

    cordy_value_t new_vec = VEC(vec.len);

    int sum = 0, value;
    for (int i = 0; i < vec.len; i++) {
        AS_INT(value, vec.ptr[i]);
        sum += value;
        new_vec.vec_value.ptr[i] = INT(sum);
        new_vec.vec_value.len++;
    }

    return new_vec;
}