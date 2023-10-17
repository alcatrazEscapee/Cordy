use cc::Build;

fn main() {
    println!("cargo:rerun-if-changed=src/main.c");
    println!("cargo:rerun-if-changed=../cordy/cordy.h");
    Build::new()
        .compiler("gcc")
        .shared_flag(true)
        .warnings(false) // Everything will be unused, so there's no point here
        .file("src/main.c")
        .compile("main")
}
