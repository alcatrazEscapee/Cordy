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

**Example**

```
>>> fn() {} is function
true
>>> print is function
true
>>> 'hello' is function
false
```

### Iterable `iterable`

The keyword `iterable` can be used in an `is` expression, to check if a value is of any `iterable` type.

**Example**

```
>>> '123' is iterable
true
>>> 123 is iterable
false
```

### Repr `repr(x: any) -> str`

Returns the full representation of `x`, as a string. Strings are wrapped in single quotes, unlike `str`, although is functionally similar in other respects.

**Example**

```
>>> repr('hello')
'hello'
```

### Eval `eval(x: str) -> any`

Compiles and evaluates the Cordy expression represented by the string `x`. This is the inverse operation of `repr`. Note that `eval` cannot reference any variables and cannot define any (unless inside an anonymous function). Raises an error if the string `x` is not valid and evaluable Cordy code.

**Example**

```
>>> '1 + 2' . eval
3
```

### Len `len(x: iterable) -> int`

Returns the length of `x`. For strings, this returns the number of Unicode Scalar Values. It is `O(1)` except for `str`, which is `O(n)`.

### Range `range(...) -> list<int>`

Possible signatures:

- `range(stop: int) -> list<int>`
- `range(start: int, stop: int) -> list<int>`
- `range(start: int, stop: int, step: int) -> list<int>`

Returns a range of `int`, from `start` inclusive, to `stop` exclusive, counting by `step`. The default value of `start` is `0`, and `step` is 1 when not provided.

**Note**: this function is lazy, and will produce elements when iterated through, i.e. by calling `list`.

### Enumerate `<T> enumerate(x: iterable<A>) -> list<vector<int, A>>`

Returns a `list` of pairs, of index and value of each element in the iterable `x`.

**Note**: this function is lazy, and will produce elements when iterated through, i.e. by calling `list`.

**Example**

```
>>> list(enumerate('hey'))
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

### Min By `min_by(...) -> int`

Possible signatures:

- `<A, B> min_by(key: fn(A) -> B, it: iterable<A>)`
- `<A> min_by(cmp: fn(A, A) -> int, it: iterable<A>)`

Returns either a minimum of `it` by the key `key`, or a minimum by the comparator function `cmp`, depending on the number of arguments required by `key` / `cmp`. Raises an error when `it` is an empty iterable.

### Max `max(...) -> int`

Possible signatures:

- `max(it: iterable<int>) -> int`
- `max(int, ...) -> int`

With one argument, returns the maximum of each value in the iterable. With more than one argument, returns the maximum of all the arguments. Raises an error when invoked with no arguments.

### Max By `max_by(...) -> int`

Possible signatures:

- `<A, B> max_by(key: fn(A) -> B, it: iterable<A>)`
- `<A> max_by(cmp: fn(A, A) -> int, it: iterable<A>)`

Returns either a maximum of `it` by the key function `key`, or a minimum by the comparator function `cmp`, depending on the number of arguments required by `key` / `cmp`. Raises an error when `it` is an empty iterable.

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

### Sort `<A> sort(it: iterable<A>) -> list<A>`

Returns a list of the elements in `it`, sorted in ascending order. Note that if `it` contains multiple different types the returned order is unspecified as different types will compare as equal.

**Example**

```
>>> [1, 5, 3, 2, 4] . sort
[1, 2, 3, 4, 5]
```

### Sort By `<A> sort_by(...) -> list<A>`

Possible signatures:

- `<A, B> sort_by(key: fn(A) -> B, it: iterable<A>) -> list<A>`
- `<A> sort_by(cmp: fn(A, A) -> int, it: iterable<A>) -> list<A>`

Returns the elements from `it` in a sorted ascending order, either by the key function `key`, or by the comparator function `cmp`, depending on the number of arguments required by `key` / `cmp`.

### Reverse `<A> reverse(it: iterable<A>) -> list<A>`

Returns a list of the elements in `it`, in reverse order.

**Example**

```
>>> [1, 3, 5, 7] . reverse
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

### Any `<A> any(f: fn(A) -> bool, it: iterable<A>) -> bool`

Returns `true` if any of the values in `it` return `true` to the function `f`. This is lazy and only evaluates as many elements in `it` as needed.

```
>>> [-1, -6, 3, -2] . any(>0)
true
```

### All `<A> all(f: fn(A) -> bool, it: iterable<A>) -> bool`

Returns `true` if all the values in `it` return `true` to the function `f`. This is lazy and only evaluates as many elements in `it` as needed.

**Example**

```
>>> [1, 6, 3, 2] . all(>0)
true
```

### Memoize `<A> memoize(f: fn(...) -> A) -> fn(...) -> A`

This creates a memorizing wrapper around a function. The returned function will cache all values based on the input parameters. The return value is invoked identically to the provided function.

**Example**

```
>>> fn add(x, y) {
...     print('add was called')
...     x + y
... }
>>> let cached_add = memoize(add)
>>> add(1, 2)
add was called
3
>>> add(1, 2)
3
```

### Pop `<A> pop(it: iterable<A>) -> A`

Pops a value from a collection. For `list`, this will be a value at the back of the collection. For a `heap`, this is the top of the heap, i.e. the minimum value. For a `dict`, this will return a key-value pair.

### Pop Front `<A> pop_front(it: list<A>) -> A`

Pops a value from the front of a list.

### Push `<A> push(x: A, it: list<A> | set<A> | heap<A>) -> iterable<A>`

Pushes a value `x` into a collection `it`. For `list`, this will be a value at the back of the collection. Returns the collection.

### Push Front `<A> push_front(x: A, it: list<A>) -> list<A>`

Pushes a value `x` into the front of a list. Returns the list.

### Insert `insert(...)`

Possible signatures:

- `<A> insert(index: int, x: A, it: list<A>) -> list<A>`
- `<K, V> insert(key: K, value: V, it: dict<K, V>) -> dict<K, V>`

Inserts a value `x` into a collection `it`, either by key, or by value. For `list`, will return an error if the index is out of bounds of the list. Returns the collection.

### Remove `remove(...)`

Possible signatures:

- `<A> remove(index: int, it: list<A>) -> A`
- `<A> remove(value: A, it: set<A>) -> bool`
- `<K, V> remove(key: K, it: dict<K, V>) -> bool`

Removes a value from a collection `it`, with the behavior differing by collection. For `list`, this removes a value by index. For `set`, this will remove by value, and return `true` if the value was present. For `dict`, this will remove an entry by key, and return `true` if the key was removed.

### Clear `clear(it: iterable) -> iterable`

Clears the contents of a collection. Returns the collection.

### Find `<A> find(x: A | fn(A) -> bool, it: iterable<A>) -> A`

If `x` is a function, this will find the first value from the left in `it` where a value returns `true` to the function. If `x` is a value, it will return the first value from the left in `it` where a value is equal to `x`.

`find(x)` is equivalent to `find(==x)` if `x` is not a function.

Returns `nil` if the value was not found.

**Example**

```
>>> [1, 2, 3, 4, 5] . find(4)
4
>>> [1, 2, 3, 4, 5] . find(fn(i) -> i % 3 == 0)
3
```

### Right Find `<A> rfind(x: A | fn(A) -> bool, it: iterable<A>) -> A`

If `x` is a function, this will find the first value from the right in `it` where a value returns `true` to the function. If `x` is a value, it will return the first value from the right in `it` where a value is equal to `x`.

`rfind(x)` is equivalent to `rfind(==x)` if `x` is not a function.

Returns `nil` if the value was not found.

**Example**

```
>>> [1, 2, 3, 4, 5] . rfind(4)
4
>>> [1, 2, 3, 4, 5] . rfind(fn(i) -> i % 3 == 0)
3
```

### Index Of `<A> index_of(x: A | fn(A) -> bool, it: iterable<A>) -> int`

Like `find`, but for an indexable collection, returns the index where the value was found, not the value itself.

### Right Index Of `<A> rindex_of(x: A | fn(A) -> bool, it: iterable<A>) -> int`

Like `rfind`, but for an indexable collection, returns the index where the value was found, not the value itself.

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

### (Str) Replace `replace(search: str, with: str, self: str) -> str`

Replaces all occurrences of `search` with `with` in `self`

**Example**

```
>>> 'bob and alice' . replace('and', 'or')
'bob or alice'
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
