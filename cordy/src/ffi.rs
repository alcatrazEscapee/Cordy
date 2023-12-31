use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use fxhash::FxBuildHasher;
use libloading::{Library, Symbol};
use cordy_sys::compiler::FunctionLibrary;
use cordy_sys::vm::{FunctionInterface, IntoValue, Type, ValuePtr, ValueResult};
use cordy_sys::vm::RuntimeError;

use RuntimeError::OSError;


/// The type of all Cordy FFI functions.
/// - First argument is a pointer to the function arguments, as an array.
/// - Second argument is the number of arguments invoked with.
type ExternFunc = unsafe extern "C" fn(*const CordyValue, usize) -> CordyValue;
type InitFunc = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);

#[repr(u32)]
#[allow(dead_code)] // Since this is primarily constructed by external code
enum CordyType {
    Err = 0,
    Nil = 1,
    Bool = 2,
    Int = 3,
    Str = 4,
    Array = 5,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CordyValueArray {
    ptr: *mut CordyValue,
    len: usize,
    capacity: usize,
}

#[repr(C)]
union CordyValueUnion {
    int: i64,
    str: *const c_char,
    vec: CordyValueArray
}

#[repr(C)]
struct CordyValue {
    ty: CordyType,
    value: CordyValueUnion
}


/// The FFI instance type that is used in the VM.
pub struct ExternalLibraryInterface {
    /// A map, provided by command line arguments, of module names -> file paths used for native modules.
    links: HashMap<String, String, FxBuildHasher>,
    /// A map of `library_id` to loaded `Library`
    libraries: HashMap<u32, Library, FxBuildHasher>,
}

impl ExternalLibraryInterface {
    pub fn new(links: HashMap<String, String, FxBuildHasher>) -> ExternalLibraryInterface {
        ExternalLibraryInterface { links, libraries: HashMap::with_hasher(FxBuildHasher::default()) }
    }
}

impl FunctionInterface for ExternalLibraryInterface {
    fn handle(&mut self, functions: &FunctionLibrary, handle_id: u32, args: Vec<ValuePtr>) -> ValueResult {
        // Find the module and method name
        let entry = functions.lookup(handle_id);

        // Try and find the library by module name, and if not present, load one after mapping through the linked libraries
        let library: &mut Library = match self.libraries.entry(entry.module_id) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let library_name = match self.links.get(&entry.module_name) {
                    Some(it) => it,
                    None => return OSError(format!("No --link provided for native module {}", entry.module_name)).err()
                };
                let library = match unsafe { Library::new(library_name) } {
                    Ok(it) => {
                        unsafe {
                            // First time initialization
                            let api_init = match it.get(b"_cordy_api_init\0") {
                                Ok(it) => it,
                                Err(e) => return OSError(format!("No _cordy_api_init() found in library '{}' for module {}: {}", library_name, entry.module_name, e)).err()
                            };
                            init(*api_init);
                        }
                        it
                    },
                    Err(e) => return OSError(format!("Loading library '{}' for module {}: {}", library_name, entry.module_name, e)).err()
                };

                e.insert(library)
            }
        };
        unsafe {
            let sym: Symbol<ExternFunc> = match library.get(entry.symbol()) {
                Ok(it) => it,
                Err(e) => return OSError(format!("Loading native function '{}' from module {}: {}", entry.method_name, entry.module_name, e)).err()
            };
            call(*sym, args)
        }
    }
}


/// Performs first-time initialization of a library, which provides Cordy API function pointers to the library
fn init(f: InitFunc) {
    unsafe { f(
        cordy_api_str as *mut c_void,
        cordy_api_vec as *mut c_void,
        cordy_api_free as *mut c_void
    ) }
}


/// Top-level API to call an external function via a common API.
fn call(f: ExternFunc, args: Vec<ValuePtr>) -> ValueResult {
    // Collect arguments here. Note that this may leak memory (i.e. via string types) in order to converse with the corresponding C API
    let args = match args.into_iter()
        .map(to_value)
        .collect::<Result<Vec<CordyValue>, String>>() {
        Ok(it) => it,
        Err(e) => return OSError(e).err()
    };

    from_value(unsafe {
        f(args.as_ptr(), args.len())
    })
}


fn to_value(ptr: ValuePtr) -> Result<CordyValue, String> {
    match ptr.ty() {
        Type::Nil => to_value_int(CordyType::Nil, 0),
        Type::Bool => to_value_int(CordyType::Bool, ptr.as_bool() as i64),
        Type::Int => to_value_int(CordyType::Int, ptr.as_precise_int()),
        Type::ShortStr | Type::LongStr => {
            match CString::new(ptr.as_str_slice()) {
                Ok(it) => Ok(to_value_str(it)),
                Err(e) => Err(format!("String is not FFI compatible : {}", e))
            }
        },
        Type::Vector => to_value_array(ptr.as_vector().borrow().iter().cloned()),
        Type::List => to_value_array(ptr.as_list().borrow().iter().cloned()),
        _ => Err(String::from("FFI Type must be a primitive nil, bool, or int")),
    }
}

fn to_value_int(ty: CordyType, int: i64) -> Result<CordyValue, String> {
    Ok(CordyValue { ty, value: CordyValueUnion { int }})
}

fn to_value_str(str: CString) -> CordyValue {
    CordyValue { ty: CordyType::Str, value: CordyValueUnion { str: str.into_raw() }}
}

fn to_value_array<I : Iterator<Item=ValuePtr>>(iter: I) -> Result<CordyValue, String> {
    let vec: Vec<CordyValue> = iter.map(to_value).collect::<Result<Vec<CordyValue>, String>>()?;
    let (ptr, len, capacity) = vec.into_raw_parts();
    Ok(CordyValue { ty: CordyType::Array, value: CordyValueUnion { vec: CordyValueArray { ptr, len, capacity } }})
}

fn from_value(value: CordyValue) -> ValueResult {
    unsafe {
        match value.ty {
            CordyType::Err => {
                // Errors must be static strings that don't require allocations or copies
                let c_str = CStr::from_ptr(value.value.str);
                match c_str.to_str() {
                    Ok(it) => OSError(String::from(it)),
                    Err(e) => OSError(format!("During handling of another error: {}", e))
                }.err()
            }
            CordyType::Nil => ValuePtr::nil().ok(),
            CordyType::Bool => (value.value.int != 0).to_value().ok(),
            CordyType::Int => value.value.int.to_value().ok(),
            CordyType::Str => {
                // Strings are passed and give ownership to Cordy
                let c_str = CString::from_raw(value.value.str as *mut c_char);
                match c_str.to_str() {
                    Ok(it) => it.to_value().ok(),
                    Err(e) => OSError(format!("During handling of another error: {}", e)).err()
                }
            },
            CordyType::Array => {
                let old = value.value.vec;
                let mut new = Vec::with_capacity(old.capacity);
                for value in Vec::from_raw_parts(old.ptr, old.len, old.capacity) {
                    new.push(from_value(value)?);
                }
                new.to_value().ok()
            }
        }
    }
}


/// Creates a `cordy_value_t` from a `const char*` when called from C
/// This does not take ownership or deallocate the provided slice.
///
/// Note that there are several invariants that **must** be checked by the caller, as errors are not handled:
/// - The string provided must be valid UTF-8
/// - Any allocations or null-byte issues must resolve
fn cordy_api_str(str: *const c_char) -> CordyValue {
    unsafe {
        let str = CStr::from_ptr(str).to_str().unwrap();
        let c_str = CString::new(str.as_bytes()).unwrap();

        to_value_str(c_str)
    }
}

fn cordy_api_vec(capacity: usize) -> CordyValue {
    let vec = Vec::with_capacity(capacity);
    let (ptr, len, capacity) = vec.into_raw_parts();
    CordyValue { ty: CordyType::Array, value: CordyValueUnion { vec: CordyValueArray { ptr, len, capacity } }}
}

/// Frees a `cordy_value_t`. The value should not be used after this.
fn cordy_api_free(value: CordyValue) {
    match value.ty {
        CordyType::Str => {
            unsafe {
                let _ = CString::from_raw(value.value.str as *mut c_char);
            }
        },
        _ => {},
    }
}


#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::ffi::c_void;
    use cordy_sys::AsError;
    use cordy_sys::util::assert_eq;
    use cordy_sys::vm::{IntoValue, ValuePtr};

    use crate::ffi;
    use crate::ffi::{CordyValue, ExternFunc};


    #[test] fn test_to_value_nil() { run(ValuePtr::nil()) }
    #[test] fn test_to_value_true() { run(true.to_value()) }
    #[test] fn test_to_value_false() { run(false.to_value()) }
    #[test] fn test_to_value_int_zero() { run(0_i64.to_value()) }
    #[test] fn test_to_value_int() { run(17892341_i64.to_value()) }
    #[test] fn test_to_value_empty_str() { run("".to_value()) }
    #[test] fn test_to_value_short_str() { run("hello".to_value()) }
    #[test] fn test_to_value_long_str() { run("hello goodbye and other words".to_value()) }

    fn run(before: ValuePtr) {
        let value = ffi::to_value(before.clone()).expect("Invalid to_value()");
        let after = ffi::from_value(value).as_result().expect("Invalid from_value()");
        assert_eq!(after, before);
    }

    // This external interface is provided by `cordy-test`, which compiles it from C
    // It is used as a test-only dependency to test the FFI works (although not the dynamic loading part)
    extern crate cordy_test;
    extern "C" {
        fn get_42(args: *const CordyValue, n: usize) -> CordyValue;
        fn add2(args: *const CordyValue, n: usize) -> CordyValue;
        fn is_nil(args: *const CordyValue, n: usize) -> CordyValue;
        fn substring(args: *const CordyValue, n: usize) -> CordyValue;
        fn partial_sums(args: *const CordyValue, n: usize) -> CordyValue;

        fn _cordy_api_init(str: *mut c_void, array: *mut c_void, free: *mut c_void);
    }

    #[test] fn test_ffi_get_42() { run_ffi(get_42, vec![], 42_i64.to_value()) }
    #[test] fn test_ffi_add2() { run_ffi(add2, vec![123_i64.to_value(), 456_i64.to_value()], 579_i64.to_value()) }
    #[test] fn test_ffi_is_nil_yes() { run_ffi(is_nil, vec![ValuePtr::nil()], true.to_value()) }
    #[test] fn test_ffi_is_nil_no_bool() { run_ffi(is_nil, vec![false.to_value()], false.to_value()) }
    #[test] fn test_ffi_is_nil_no_int() { run_ffi(is_nil, vec![123_i64.to_value()], false.to_value()) }
    #[test] fn test_ffi_substring() { run_ffi(substring, vec!["substring time".to_value(), 3_i64.to_value(), 6_i64.to_value()], "string".to_value()) }
    #[test] fn test_ffi_partial_sums_empty_vector() { run_ffi(partial_sums, vec![Vec::new().to_value()], vec![].to_value()) }
    #[test] fn test_ffi_partial_sums_empty_list() { run_ffi(partial_sums, vec![VecDeque::new().to_value()], vec![].to_value()) }
    #[test] fn test_ffi_partial_sums_some() { run_ffi(partial_sums, vec![vec![10_i64.to_value(), 25_i64.to_value(), 3_i64.to_value()].to_value()], vec![10_i64.to_value(), 35_i64.to_value(), 38_i64.to_value()].to_value()) }

    #[test] fn test_err_add2_too_few_args() { run_err(add2, vec![], "OsError: n == 2\n  at: line 11 (external: src/main.c)") }
    #[test] fn test_err_add2_first_arg_not_int() { run_err(add2, vec![ValuePtr::nil(), 456_i64.to_value()], "OsError: args[0].ty == TY_INT\n  at: line 12 (external: src/main.c)") }
    #[test] fn test_err_add2_second_arg_not_int() { run_err(add2, vec![123_i64.to_value(), false.to_value()], "OsError: args[1].ty == TY_INT\n  at: line 14 (external: src/main.c)") }
    #[test] fn test_err_is_nil_too_few_args() { run_err(is_nil, vec![], "OsError: n == 1\n  at: line 19 (external: src/main.c)") }
    #[test] fn test_err_partial_sums_no_args() { run_err(partial_sums, vec![], "OsError: n == 1\n  at: line 48 (external: src/main.c)") }
    #[test] fn test_err_partial_sums_not_a_vector() { run_err(partial_sums, vec![123_i64.to_value()], "OsError: args[0].ty == TY_ARRAY\n  at: line 49 (external: src/main.c)") }
    #[test] fn test_err_partial_sums_vector_contains_bool() { run_err(partial_sums, vec![vec![10_i64.to_value(), true.to_value()].to_value()], "OsError: vec.ptr[i].ty == TY_INT\n  at: line 55 (external: src/main.c)") }


    fn run_ffi(f: ExternFunc, args: Vec<ValuePtr>, expected: ValuePtr) {
        ffi::init(_cordy_api_init);
        assert_eq!(ffi::call(f, args).as_result().expect("FFI returned error"), expected);
    }

    fn run_err(f: ExternFunc, args: Vec<ValuePtr>, expected: &'static str) {
        ffi::init(_cordy_api_init);
        assert_eq(ffi::call(f, args).as_result().expect_err("Expected FFI to return error").as_err().as_error(), String::from(expected));
    }
}
