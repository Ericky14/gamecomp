//! Build script for gamecomp.
//!
//! Compiles GLSL compute shaders to SPIR-V at build time using shaderc.
//! The compiled SPIR-V binaries are included in the binary via `include_bytes!`.

use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/compositor/shaders/");
    println!("cargo:rerun-if-changed=protocols/wayland-drm.xml");

    let shader_dir = Path::new("src/compositor/shaders");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir);

    // Find all .comp files in the shaders directory.
    let shader_files: Vec<_> = fs::read_dir(shader_dir)
        .expect("failed to read shader directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("comp") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if shader_files.is_empty() {
        eprintln!(
            "warning: no .comp shader files found in {}",
            shader_dir.display()
        );
        return;
    }

    // Initialize shaderc compiler.
    let compiler = shaderc::Compiler::new().expect("failed to create shaderc compiler");
    let mut options =
        shaderc::CompileOptions::new().expect("failed to create shaderc compile options");

    // Set optimization level.
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    options.set_target_spirv(shaderc::SpirvVersion::V1_6);

    for shader_path in &shader_files {
        let shader_name = shader_path.file_stem().unwrap().to_str().unwrap();
        let source = fs::read_to_string(shader_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", shader_path.display(), e));

        eprintln!("compiling shader: {}", shader_path.display());

        let artifact = compiler
            .compile_into_spirv(
                &source,
                shaderc::ShaderKind::Compute,
                shader_name,
                "main",
                Some(&options),
            )
            .unwrap_or_else(|e| panic!("shader compilation failed for {}: {}", shader_name, e));

        if artifact.get_num_warnings() > 0 {
            eprintln!(
                "shader warnings for {}: {}",
                shader_name,
                artifact.get_warning_messages()
            );
        }

        // Write SPIR-V binary.
        let spv_path = out_path.join(format!("{}.spv", shader_name));
        fs::write(&spv_path, artifact.as_binary_u8())
            .unwrap_or_else(|e| panic!("failed to write {}: {}", spv_path.display(), e));

        eprintln!("compiled {} -> {}", shader_name, spv_path.display());
    }
}
