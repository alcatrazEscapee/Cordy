#ifndef CORDY_H
#define CORDY_H

#include <stdbool.h>
#include <stdint.h>

// Injected API Calls
// When Cordy first initializes a dynamic library it will inject these as function calls back into Cordy API code
static void* _cordy_api_str = 0;
static void* _cordy_api_vec = 0;
static void* _cordy_api_free = 0;

void _cordy_api_init(void* str, void* vec, void* free) {
    _cordy_api_str = str;
    _cordy_api_vec = vec;
    _cordy_api_free = free;
}

/**
 * An enum representing the possible types of a Cordy FFI value. It is used as the discriminator in the `cordy_value_t` union.
 * Wherever possible, avoid using this directly, and instead use the `NIL()`, `BOOL()`, etc. macros to create new values, and the `AS_NIL()`, `AS_BOOL()`, etc. macros to parse existing values.
 */
#ifdef __cplusplus
enum cordy_type_t : uint32_t {
#else
typedef uint32_t cordy_type_t;
enum {
#endif
    TY_ERR = 0,
    TY_NIL = 1,
    TY_BOOL = 2,
    TY_INT = 3,
    TY_STR = 4,
    TY_ARRAY = 5,
};


#ifdef __cplusplus
extern "C" {
#endif

struct cordy_value_s;

typedef const char* str_t;

/**
 * A vector type exposed to C FFI code. This is a deconstructed `Vec<>` from Rust.
 * - The pointer, capacity must not be changed
 * - The user must manage everything about the vec themselves
 */
typedef struct {
    struct cordy_value_s * ptr;
    size_t len;
    size_t capacity;
} cordy_vec_t;


/**
 * The Cordy FFI value type. All values passed from Cordy to C/C++ code and vice versa must be of this type.
 * Cordy FFI functions must have the following signature:
 * ```c
 * cordy_value_t my_ffi_function(cordy_value_t* args, size_t n);
 * ```
 * Where `args` is a pointer to an array of `n` arguments to the FFI function.
 * **Important**: The FFI function **must not** attempt to free the `args` pointer.
 */
typedef struct cordy_value_s {
    cordy_type_t ty;
    union {
        bool bool_value;
        int64_t int_value;
        str_t str_value;
        cordy_vec_t vec_value;
    };
} cordy_value_t;


// Macro that defines the API for possible Cordy FFI functions.
// The first argument is a pointer to the function arguments, and `n` is the number of arguments.
#define CORDY_EXTERN(name) cordy_value_t name(cordy_value_t* args, size_t n)

#define _VALUE(type, value_field, value) ((cordy_value_t) { .ty = type, .value_field = value })

// Macros that construct a new cordy_value_t
#define ERR(x) ERR(#x)
#define _ERR(x) _VALUE(TY_ERR, str_value, x)
#define NIL() _VALUE(TY_NIL, int_value, 0)
#define BOOL(x) _VALUE(TY_BOOL, bool_value, x)
#define INT(x) _VALUE(TY_INT, int_value, x)
#define STR(x) (((cordy_value_t (*) ( const char* )) _cordy_api_str)(x))
#define VEC(n) (((cordy_value_t (*) ( size_t )) _cordy_api_vec)(n))

#define FREE(x) (((void (*) ( cordy_value_t )) _cordy_api_free) (x))

/**
 * Basic assertion, that can be used in a method which returns `cordy_value_t`. Usage:
 * ```
 * ASSERT(n == 3);
 * ```
 */
#define ASSERT(x) _ASSERT_S(x, #x)
#define _ASSERT_S(x, err) _ASSERT(x, err "\n  at: line " LITERAL(__LINE__) " (external: " __FILE__ ")")
#define _ASSERT(x, err) do { \
    if (!(x)) {              \
        return _ERR(err);    \
    }                        \
} while (0)

/**
 * Used to call other methods which return a `cordy_value_t` and check the result for errors.
 * - `name` may be either `cordy_value_t <name>`, in which case this will declare the variable, or `<name>` to reuse an existing variable.
 *
 * Usage:
 * ```
 * TRY(cordy_value_t ret, some_other_method()); // declares and assigns to `ret`
 * TRY(ret, some_other_method()); // only assigns to `ret`
 * ```
 */
#define TRY(name, x)           \
ONLY_DECL(name;)               \
do {                           \
    cordy_value_t __try = (x); \
    if (__try.ty == TY_ERR) {  \
        return __try;          \
    }                          \
    STRIP_DECL(name) = __try;  \
} while (0)

#define _AS(type, value_field, name, x) \
ONLY_DECL(name;)                                    \
do {                                                \
    cordy_value_t __as = (x);                       \
    _ASSERT_S(__as.ty == type, #x ".ty == " #type); \
    STRIP_DECL(name) = __as.value_field;            \
} while (0)

// Macros that convert a cordy_value_t into a domain type
/**
 * Macros that convert a cordy_value_t into a domain type, or returns an error.
 * Note these will optionally declare a value of the given type - which must be `bool` or `int64_t`
 * Usage:
 * ```
 * AS_INT(int64_t x, args[0]) // declares `int64_t x`, and assigns if `args[0]` is an int, otherwise returns an error
 * AS_INT(x, args[0]) // does not declare `x`, otherwise same as above
 * ```
 *
 * Note that for `nil` values there is instead only `IS_NIL()` which simply returns a `bool` if the value is equal to `nil`
 * It can be used as `ASSERT(IS_NIL(x))` to achieve the same effect as `AS_NIL()` would've done.
 */
#define IS_NIL(x) (x.ty == TY_NIL)
#define AS_BOOL(name, x) _AS(TY_BOOL, bool_value, name, x)
#define AS_INT(name, x) _AS(TY_INT, int_value, name, x)
#define AS_STR(name, x) _AS(TY_STR, str_value, name, x)
#define AS_VEC(name, x) _AS(TY_ARRAY, vec_value, name, x)


// where xy is a `cordy_value_t <name>`, should return just `<name>
// where xy is just `<name>`, return input unaltered
#define STRIP_DECL(xy) _REFLECT(_ARG2, _DECL_ ## xy , xy)
#define ONLY_DECL(xy) _REFLECT(_ARG3, _DECL_ ## xy , xy ,)

#define _DECL_cordy_value_t _DECL_
#define _DECL_cordy_vec_t _DECL_
#define _DECL_bool _DECL_
#define _DECL_int64_t _DECL_
#define _DECL_str_t _DECL_

#define _DECL_ ~ ,

// Internal Macro Helpers
#define _ARG2(x1, x2, ...) x2
#define _ARG3(x1, x2, x3, ...) x3
#define _REFLECT(m, ...) _EXPAND(m _WRAP(__VA_ARGS__))
#define _EXPAND(x) x
#define _WRAP(...) ( __VA_ARGS__ )

#define LITERAL(x) _LITERAL(x)
#define _LITERAL(x) # x

#ifdef __cplusplus
};
#endif

#endif //CORDY_H