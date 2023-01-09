# Cordy Standard Library

The Cordy standard library consists of a number of functions which are placed in the global namespace. These are not shadow-able or able to be overriden.


### Type Signatures

The below type signatures are entirely for documentation purposes, as Cordy does not have explicit type signatures and is entirely dynamically typed. The type signatures below obey the following conventions:

- `any` refers to any type.
- `iterable<T>` refers to an iterable type with element type `T`, which includes `list<T>`, `set<T>`, `dict<K, V>` (where `T = vector<K, V>`), `heap<T>`, `vector<T>`, or `str` (where `T = str`)
- `fn(A1, A2, ...) -> B` refers to a function which accepts arguments of types `A1, A2, ...`, and returns a value of type `B`
- `...` is used to indicate a function has multiple possible signatures, which will be noted below.
- `f(T, ...)` is used to indicate the function `f` can take any number of arguments of the type `T`
- `|` is used to indicate an argument can be one of multiple types, i.e. `it: A | B` indicates `it` can be of type `A` or `B`.

---


### Print `print(any, ...)`

Prints each argument, space seperated and with a single `\n` suffix, to standard output. Non-string types will have `str()` called on them before printing.

When called with no arguments, prints a single `\n` character.

Returns `nil`

### Bool `bool(x: any) -> bool`

Returns the argument as a boolean. `nil`, `0`, `false`, `''`, and empty collections, will return `false`, everything else will return `true`.

The keyword `bool` can also be used in an `is` expression, to check if a value is of the type `bool`.

### Int `int(x: any) -> int`

Returns the argument as an integer. `nil` and `false` evaluate to `0`, where strings will be parsed as an integer or raise an error.

The keyword `int` can also be used in an `is` expression, to check if a value is of the type `int`.

**Example**

```
>>> int('3')
3
>>> 3 is int
true
>>> '3' is int
false
```

### Str `str(x: any) -> str`

Returns the argument as a string. See also `repr`.

The keyword `str` can also be used in an `is` expression, to check if a value is of the type `str`.


### List `list(...) -> list`

Possible signatures:

- `list() -> list`
- `<T> list(it: iterable<T>) -> list<T>`
- `<T> list(T, ...) -> list<T>`

With no arguments, creates an empty list, the same as `[]`. With one argument, treats the argument as an iterable and copies each element into a new list. With more than one argument, collects each argument into a list.

The keyword `list` can also be used in an `is` expression, to check if a value is of the type `list`.

**Example**

```
>>> list()
[]
>>> list('hello')
['h', 'e', 'l', 'l', 'o']
>>> list(1, 2, 3, 4)
[1, 2, 3, 4]
```

### Set `set(...) -> set`

Possible signatures:

- `set() -> set`
- `<T> set(it: iterable<T>) -> set<T>`
- `<T> set(T, ...) -> set<T>`

With no arguments, creates an empty set. With one argument, treats the argument as an iterable and copies each element into a new set. With more than one argument, collects each argument into a set.

The keyword `set` can also be used in an `is` expression, to check if a value is of the type `set`.

**Example**

```
>>> set()
{}
>>> set('blahaj')
{'b', 'l', 'h', 'a', 'j'}
>>> set(1, 2, 3, 4)
{1, 2, 3, 4}
```

### Dict `dict(...) -> dict`

Possible signatures:

- `dict() -> dict`
- `<K, V> dict(it: iterable<vector<K, V>>) -> dict<K, V>`
- `<K, V> dict(vector<K, V>, ...) -> dict<K, V>`

With no arguments, creates an empty dictionary. With one argument, treats the argument as an iterable of key-value pairs and collects it into a new dictionary. With more than one argument, treats each argument as a key-value pair and collects each argument into a dictionary.

The keyword `dict` can also be used in an `is` expression, to check if a value is of the type `dict`.

### Heap `heap(...) -> heap`

Possible signatures:

- `heap() -> heap`
- `<T> heap(it: iterable<T>) -> heap<T>`
- `<T> heap(T, ...) -> heap<T>`

With no arguments, creates an empty heap. With one argument, treats the argument as an iterable and copies each element into a new heap, maintaining the heap invariant. With more than one argument, collects each argument into a heap, maintaining the heap invariant.

**Note:** Heaps of different types behavior is unspecified, as different types will compare equal and can have any internal ordering.

The keyword `heap` can also be used in an `is` expression, to check if a value is of the type `heap`.

### Vector `vector(...) -> vector`

Possible signatures:

- `vector() -> vector`
- `<T> vector(it: iterable<T>) -> vector<T>`
- `<T> vector(T, ...) -> vector<T>`

With no arguments, creates an empty vector. With one argument, treats the argument as an iterable and copies each element into a new vector. With more than one argument, collects each argument into a vector.

The keyword `vector` can also be used in an `is` expression, to check if a value is of the type `vector`.

### Function `function`

The keyword `function` can be used in an `is` expression, to check if a value is of the type `function`.

```
>>> fn() {} is function
true
>>> print is function
true
>>> 'hello' is function
false
```

### Repr `repr(x: any) -> str`

Returns the full representation of `x`, as a string. Strings are wrapped in single quotes, unlike `str`, although is functionally similar in other respects.

```
>>> repr('hello')
'hello'
```

### Len `len(x: iterable) -> int`

Returns the length of `x`. For strings, this returns the number of Unicode Scalar Values. It is `O(1)` except for `str`, which is `O(n)`.

### Range `range(...) -> list<int>`

Possible signatures:

- `range(stop: int) -> list<int>`
- `range(start: int, stop: int) -> list<int>`
- `range(start: int, stop: int, step: int) -> list<int>`

Returns a list of `int`, from `start` inclusive, to `stop` exclusive, counting by `step`. The default value of `start` is `0`, and `step` is 1 when not provided.

**Note:** When used in `for x in range()` loop, this function elides creating a `list<int>` and instead lazily populates `x`.

### Enumerate `<T> enumerate(x: iterable<A>) -> list<vector<int, A>>`

Returns a `list` of pairs, of index and value of each element in the iterable `x`.

**Example**

```
>>> enumerate('hey')
[(0, 'h'), (1, 'e'), (2, 'y')]
```

### Sum `sum(...) -> int`

Possible signatures:

- `sum(it: iterable<int>) -> int`
- `sum(int, ...) -> int`

With one argument, returns the sum of each value in the iterable. With more than one argument, returns the sum of all the arguments. Raises an error when invoked with no arguments.

### Min `min(...) -> int`

Possible signatures:

- `min(it: iterable<int>) -> int`
- `min(int, ...) -> int`

With one argument, returns the minimum of each value in the iterable. With more than one argument, returns the minimum of all the arguments. Raises an error when invoked with no arguments.

### Max `max(...) -> int`

Possible signatures:

- `max(it: iterable<int>) -> int`
- `max(int, ...) -> int`

With one argument, returns the maximum of each value in the iterable. With more than one argument, returns the maximum of all the arguments. Raises an error when invoked with no arguments.

### Map `<A, B> map(f: fn(A) -> B, it: iterable<A>) -> list<B>`

Applies the function `f` to each value in the iterable `it`, and returns the list of each result.

**Example**

```
>>> [1, 3, 5] . map(+1)
[2, 4, 6]
```

### Filter `<A> filter(f: fn(A) -> any, it: iterable<A>) -> list<A>`

Applies the function `f` to each value in the iterable `it`, and retains that value if it returns a truthy value. Returns a list of all elements which returned a truthy value.

**Example**

```
>>> [-2, 4, -4, 2] . filter(>0)
[4, 2]
```

### Flat Map `<A, B> flat_map(f: fn(A) -> iterable<B>, it: iterable<A>) -> list<B>`

Applies the function `f` to each element in `it`, and then concatenates the results. This is equivalent to `. map(f) . concat`.

**Example**

```
>>> [1, 2, 3, 4] . flat_map(fn(i) -> range(i))
[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
```

### Concat `<A> concat(it: iterable<iterable<A>>) -> list<A>`

Concatenates the iterables in the input into one list. This is equivalent to `flat_map(fn(x) -> x)`, but should be preferred over that due to performance.

**Example**

```
>>> [[1, 2, 3], [4, 5, 6], [7, 8, 9]] . concat
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Zip `<A> zip(...) -> list<A>`

Possible signatures:

- `<A> zip(it: iterable<iterable<A>>) -> list<A>`
- `<A> zip(iterable<A>, ...) -> list<A>`

When invoked with a single argument, treats the argument as an iterable and each element as an individual argument. Then, iterates each iterable in parallel, returning a list of vectors until the shortest iterable is exhausted.

**Example**

```
>>> zip([1, 2, 3], [10, 20, 30])
[(1, 10), (2, 20), (3, 30)]
>>> zip(['hello', 'the', 'world'])
[('h', 't', 'w'), ('e', 'h', 'o'), ('e', 'e', 'r')]
```

### Reduce `<A> reduce(f: fn(A, A) -> A, it: iterable<A>) -> A`

Reduces an iterable to a single value by successively applying `f` on the first two elements in the iterable, until only one remains. Raises an error if the argument was an empty iterable

**Example**

```
>>> [1, 3, 5, 7] . reduce(+)
16
>>> ['hello', 'the', 'world'] . reduce(fn(a, b) -> a + ' ' + b)
'hello the world'
```

### Sorted `<A> sorted(it: iterable<A>) -> list<A>`

Returns a list of the elements in `it`, sorted in ascending order. Note that if `it` contains multiple different types the returned order is unspecified as different types will compare as equal.

**Example**

```
>>> [1, 5, 3, 2, 4] . sorted
[1, 2, 3, 4, 5]
```

### Reversed `<A> reversed(it: iterable<A>) -> list<A>`

Returns a list of the elements in `it`, in reverse order.

**Example**

```
>>> [1, 3, 5, 7] . reversed
[7, 5, 3, 1]
```

### Permutations `<A> permutations(n: int, it: iterable<A>) -> list<vector<A>>`

Returns a list of all permutations of `n` elements from `it`. If `n` is larger than the length of `it`, nothing will be returned. Raises an error if `n` is negative.

**Example**

```
>>> [1, 2, 3] . permutations(2)
[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
```

### Combinations `<A> combinations(n: int, it: iterable<A>) -> list<vector<A>>`

Returns a list of all combinations of `n` elements from `it`. If `n` is larger than the length of `it`, nothing will be returned. Raises an error if `n` is negative.

**Example**

```
>>> [1, 2, 3] . combinations(2)
[(1, 2), (1, 3), (2, 3)]
```

### Pop `<A> pop(it: list<A> | set<A> | heap<A> | dict<A, ?>) -> A`

Pops a value from a collection. For `list`, this will be a value at the back of the collection. For a `heap`, this is the top of the heap, i.e. the minimum value.

### Push `<A> push(x: A, it: list<A> | set<A> | heap<A>) -> iterable<A>`

Pushes a value `x` into a collection `it`. For `list`, this will be a value at the back of the collection.

### Last `<A> last(it: list<A> | set<A>) -> A`

Returns the last element of `it`.

### Head `<A> head(it: list<A> | set<A> | heap<A>) -> A`

Returns the first element (for `list`, the front, for `heap`, the top, i.e. the minimum) of `it`.

### Init `<A> init(it: list<A> | set<A>) -> list<A>`

Returns a list of all elements of `it` except the last.

### Tail `<A> tail(it: list<A> | set<A>) -> list<A>`

Returns a list of all elements of `it` except the first.

### (Int) Abs `abs(x: int) -> int`

Returns the absolute value of `x`.

### (Int) Sqrt `sqrt(x: int) -> int`

Returns the positive integer square root of `x`, or the largest `y` such that `y*y <= x`.

### Lcm `lcm(...) -> int`

Possible signatures:

- `lcm(it: iterable<int>) -> int`
- `lcm(int, ...) -> int`

With one argument, returns the least common multiple of each value in the iterable. With more than one argument, returns the least common multiple of all the arguments. Raises an error when invoked with no arguments.

### Gcd `gcd(...) -> int`

Possible signatures:

- `gcd(it: iterable<int>) -> int`
- `gcd(int, ...) -> int`

With one argument, returns the greatest common divisor of each value in the iterable. With more than one argument, returns the greatest common divisor of all the arguments. Raises an error when invoked with no arguments.

### (Str) Split `split(delim: str, string: str) -> list<str>`

Splits `string` on the delimiter `delim` and returns a `list` of all split elements.

**Example**

```
>>> 'hello the world' . split(' ')
['hello', 'the', 'world']
```

### (Dict) Default `<K, V> default(x: V, it: dict<K, V>) -> dict<K, V>`

Sets the default value of `it` to `x`, and then returns `it`. This means that any future queries into `it` via the index syntax, if the key is not in the dictionary, will return `x`.

**Note:** If `x` is a mutable value, such as a list, the same instance will be returned from each access to the default value.

**Example**

```
let d = dict()
d['hello'] // will raise an error
d . default('nope')
d['hello'] // returns 'nope'
```

### (Dict) Keys `<K, V> keys(it: dict<K, V>) -> set<K>`

Returns a set of all keys in `it`, maintaining insertion order.

### (Dict) Values `<K, V> values(it: dict<K, V>) -> list<V>`

Returns a list of all values in `it`, maintaining insertion order.

### Find `<A> find(x: A | fn(A) -> bool, it: iterable<A>) -> int`

If `x` is a function, this will find the first index from the left in `it` where a value returns `true` to the function. If `x` is a value, it will return the first index from the left in `it` where a value is equal to `x`.

`find(x)` is equivalent `find(==x)` if `x` is not a function.

Returns `-1` if the value was not found.

**Example**

```
>>> [1, 2, 3, 4, 5] . find(4)
3
>>> [1, 2, 3, 4, 5] . find(fn(i) -> i % 3 == 0)
2
```

### Right Find `<A> rfind(x: A | fn(A) -> bool, it: iterable<A>) -> int`

If `x` is a function, this will find the first index from the right in `it` where a value returns `true` to the function. If `x` is a value, it will return the first index from the right in `it` where a value is equal to `x`.

`rfind(x)` is equivalent `rfind(==x)` if `x` is not a function.

Returns `-1` if the value was not found.

**Example**

```
>>> [1, 2, 3, 4, 5] . rfind(4)
3
>>> [1, 2, 3, 4, 5] . rfind(fn(i) -> i % 3 == 0)
2
```

### Find Count `<A> findn(x: A | fn(A) -> bool, it: iterable<A>) -> int`

If `x` is a function, this will count the number of occurrences in `it` where `x` returns `true`. If `x` is a value, it will return the number of instances of `x` in `it`.

`. findn(f)` is equivalent to `. filter(f) . len` or `. filter(==f) . len`, depending on if `f` is a function.