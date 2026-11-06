// macOS: the prebuilt ONNX Runtime (CoreML provider) references `___isPlatformVersionAtLeast`, which lives in
// clang's compiler-rt builtins. Rust links with -nodefaultlibs, so add that library explicitly.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") && std::env::var("CARGO_FEATURE_LOCAL_EMBED").is_ok() {
        if let Ok(out) = std::process::Command::new("xcrun").args(["clang", "--print-resource-dir"]).output() {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let lib = std::path::Path::new(&dir).join("lib").join("darwin");
            if lib.join("libclang_rt.osx.a").exists() {
                println!("cargo:rustc-link-search=native={}", lib.display());
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
