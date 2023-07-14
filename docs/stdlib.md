# Cordy Standard Library

The Cordy standard library consists of a number of functions which are placed in the global namespace. These are not shadow-able or able to be overridden.


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

Prints each argument, space separated and with a single `\n` suffix, to standard output. Non-string types will have `str()` called on them before printing.

When called with no arguments, prints a single `\n` character.

Returns `nil`

### Read `read() -> str`

Reads from `stdin` until end of file. Returns the result as a string.

N.B When reading from external sources, newline `\r\n` sequences will be replaced with a single `\n`.

### Read Line `read_line() -> str`

Reads a single line from `stdin`. Returns the result as a string, with the newline suffix removed.

### Read Text `read_text(path: str) -> str`

Reads from a text file, located at `path`. Any error reading the file will cause the program to exit. Returns the result as a string. 

N.B When reading from external sources, newline `\r\n` sequences will be replaced with a single `\n`.

### Write Text `write_text(path: str, content: str) -> str`

Writes the string `content` to the file at `path`, in overwrite mode. A file will be created if it does not exist, and if it does it will be overwritten.

### Env `env(...) -> any`

Possible signatures:

- `env(): dict<str, str>`
- `env(key: str) -> str | nil`

When invoked with no arguments, will return a dictionary of all currently present environment variables. When invoked with one argument, will query that specific environment variable and return its value if defined, or `nil` if not.

Prefer using `env(key)` over `env()[key]`.

**Example**

```bash
$ cat example.cor
env('DUCKS') . repr . print

$ cordy example.cor
nil
$ DUCKS=1337 cordy example.cor
'1337'
```

### Argv `argv() -> list<str>`

Returns the list of user defined program arguments when invoked. These are arguments appended to the cordy invocation after the file name.

**Example**

```bash
$ cat example.cor
argv() . repr . print

$ cordy example.cor --number-of-ducks 2 -run
['--number-of-ducks', '2', '-run']
```

### Bool `bool(x: any) -> bool`

Returns the argument as a boolean. `nil`, `0`, `false`, `''`, and empty collections, will return `false`, everything else will return `true`.

The keyword `bool` can also be used in an `is` expression, to check if a value is of the type `bool`.

### Int `int(...) -> int`

**Possible Signatures**

- `int(x: any) -> int`
- `<T> int(x: any, default: T) -> int | T`

Returns the argument as an integer. `nil` and `false` evaluate to `0`, where strings will be parsed as an integer or raise an error. If a second argument is provided, will instead return the default value instead of raising an error.

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

### Type Of `typeof(x: any) -> any`

Returns the *type* of the argument. This returns either the value, or a function representing the type for each individual input. For example, `typeof(3)` will return the function `int`, `typeof([1, 2, 3])` will return the native function `list`. The `typeof` function has a few fundamental guarantees:

1. The return value of this function will **always** be comparable using `==` to distinguish different types. That is, if `typeof(x) == typeof(y)`, these objects are of the exact same underlying type.
2. The expression `x is typeof(x)` will **always** be `true`, for any value of `x`.
3. Note that `x is y` **does not** imply that `typeof(x) == y`. This assumption is broken by some types which are considered subtypes of another:
    - `bool` is a subtype of `int`, and `int` is a subtype of `complex`.
    - Collection types, and `str` are subtypes of `iterable`
    - All types are a subtype of `any`

**Example**

```
>>> typeof(nil)
nil
>>> typeof('hello')
str
>>> typeof([1, 2, 3])
list
>>> typeof(fn() -> nil)
function
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

Note that the special case `min(int)` or `int.min` will return the lowest possible signed 64-bit integer representable.

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

Note that the special case `max(int)` or `int.max` will return the highest possible signed 64-bit integer representable.

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

### Group By <T, K> group_by(by: int | fn(T) -> K, it: iterable<T>) -> list<vector<T>> | dict<K, vector<T>>

Possible signatures:

- `<T> group_by(by: int, it: iterable<T>) -> list<vector<T>>`
- `<T, K> group_by(by: fn(T) -> K, it: iterable<T>) -> dict<K, vector<T>>`

When invoked with an int as the argument to `by`, this will return a list of groups (vectors) of length `by` from `it`, until it is exhausted. If the length of `it` does not perfectly divide `by`, the last group will contain the remainder.

**Example**

```
>>> [1, 2, 3, 4, 5, 6] . group_by(2)
[(1, 2), (3, 4), (5, 6)]
>>> [1, 2, 3, 4, 5] . group_by(3)
[(1, 2, 3), (4, 5)]
```

When invoked with a function as the argument to `by`, this will instead use the function on each element of the iterable as a key extractor. It will then create a dictionary mapping each key to its value.

**Example**

```
>>> [1, 2, 3, 4, 5] . group_by(%3)
{1: (1, 4), 2: (2, 5), 0: (3)}
>> [1, 2, 3, 4, 5] . group_by(fn(x) -> if x % 2 == 0 then 'even' else 'odd')
{'odd': (1, 3, 5), 'even': (2, 4)}
```

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

### (Int) Count Ones `count_ones(x: int) -> int`

Returns the number of ones in the 64-bit, signed, binary representation of `x`

### (Int) Count Zeros `count_zeros(x: int) -> int`

Returns the number of zeros in the 64-bit, signed, binary representation of `x`

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

### (Str) Split `split(pattern: str, x: str) -> list<str>`

Splits a string `x` based on `pattern`, with regular expression (regex) support. If `pattern` is an empty string, this functions identical to `list` applied to a string, and returns a list of all characters in the string. Otherwise, it will treat `pattern` like a regex, and split the string on sequences matching the regex.

Regex syntax is the same as used by the `replace`, and `search` functions.

**Example**

```
>>> 'abc' . split('')
['a', 'b', 'c']
>>> 'hello the world' . split(' ')
['hello', 'the', 'world']
>>> '   hello  \t the \n\n   world  !' . trim . split('\s+')
['hello', 'the', 'world', '!']
```

### (Str) Join `join(joiner: str, iter: iterable<any>) -> str`

Joins an iterable into a single string. First calls `str()` on any arguments, and joins them seperated by `joiner`.

**Example**

```
>>> [1, 2, 3, 4, 5] . join(' + ')
'1 + 2 + 3 + 4 + 5'
>>> 'hello' . join(' ')
'h e l l o'
```

### (Str) Replace `replace(...) -> str`

**Possible Signatures**

- `replace(pattern: str, replacer: str, x: str)`
- `replace(pattern: str, replacer: fn(vector<str>) -> str, x: str)`

Performs string replacement, with regular expression (regex) support. Regex syntax is the same as used by the [Fancy Regex](https://docs.rs/fancy-regex/0.11.0/fancy_regex/) crate. Additionally, `\r`, `\n`, and `\t` sequences are supported as part of a regex, and they will be replaced in the pattern with `\\r`, `\\n`, and `\\t` respectively.

When `replacer` is a string, this will replace all instances of `pattern` in the string `x` with `replacer`. When `replacer` is a function with one defined argument, this will invoke that function for each replacement to be made, to provide the result. This takes an argument of the match, which is a vector consisting of the full text, followed by any capture groups found.

Note that capture groups can also be referenced in a string via `$<number>` syntax, where `$0` represents the entire match.

**Examples**

```
>>> 'bob and alice' . replace('and', 'or')
'bob or alice'
>>> 'bob and alice' . replace('\sa', ' Ba')
'bob Band Balice'
>>> 'bob and alice' . replace('[a-z]+', '$0!')
'bob! and! alice!'
>>> 'bob and alice' . replace('[a-z]+', fn((g, *_)) -> g . reverse . reduce(+))
'bob dna ecila'
>>> 'bob and alice' . replace('([a-z])([a-z]+)', fn((_, g1, g2)) -> to_upper(g1) + g2)
'Bob And Alice'
```

### (Str) Search `search(pattern: str, x: str) -> list<vector<str>>`

Matches a string `x` against a given `pattern`, and returns a list of all results. The pattern is a regular expression (regex), with syntax identical to using `replace`. When invoked, this returns a list of all matches in the string, or an empty list if no matches are found. A match consists of a vector of all capture groups, with the first group containing the entire match.

Note, for simple substring searching, it is sufficient to test for truthiness, as no match will return an empty list. Using characters such as `^` and `$` in the regex will also ensure that only one match is possible, and so can only return a list with at most one element.

**Examples**

```
>>> 'bob and alice' . search('bob')
[('bob')]
>>> 'bob and alice' . search('a[a-z]+')
[('and'), ('alice')]
>>> 'bob and alice' . search('([a-z])[^ ]+([a-z])')
[('bob', 'b', 'b'), ('and', 'a', 'd'), ('alice', 'a', 'e')]
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
