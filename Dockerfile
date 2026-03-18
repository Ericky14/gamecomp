FROM rust:1.93

RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libclang-dev \
    libdrm-dev \
    libgbm-dev \
    libinput-dev \
    libseat-dev \
    libudev-dev \
    libwayland-dev \
    libvulkan-dev \
    libxkbcommon-dev \
    libdisplay-info-dev \
    libsystemd-dev \
    libshaderc-dev \
    glslang-dev \
    && rm -rf /var/lib/apt/lists/*

# Debian's libshaderc_combined.a is incomplete (missing glslang/SPIRV-Tools).
# The shared library works but is named libshaderc.so, not libshaderc_shared.so
# which is what shaderc-sys expects. Symlink so it finds the dynamic library.
RUN ln -s /usr/lib/x86_64-linux-gnu/libshaderc.so \
    /usr/lib/x86_64-linux-gnu/libshaderc_shared.so

# Taskfile support
RUN curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | bash \
    && apt-get install -y task

RUN rustup component add clippy rustfmt
