%global app_name gamecomp
%global debug_package %{nil}

Name:           %{app_name}
Version: 1.1.1
Release:        0%{?dist}
Summary:        High-performance single-app fullscreen Wayland compositor

License:        GPL-3.0-only
URL:            https://github.com/playtron-os/gamecomp
Source0:        %{name}-%{_arch}.tar.gz

# Disable automatic dependency detection for Rust binaries
AutoReqProv:    no

Requires:       libdrm
Requires:       mesa-libgbm
Requires:       libseat
Requires:       systemd-libs
Requires:       libwayland-server

%description
gamecomp is a high-performance single-app fullscreen Wayland compositor
designed for gaming. It provides direct DRM/KMS output with Vulkan compute
shader composition, keyboard hotplug, and session switching.

%prep
%autosetup -n %{name} -p1

%build

%install
install -Dm755 usr/bin/%{app_name} \
    %{buildroot}%{_bindir}/%{app_name}

install -Dm644 usr/share/%{app_name}/LICENSE \
    %{buildroot}%{_datadir}/%{app_name}/LICENSE

%files
%{_bindir}/%{app_name}
%{_datadir}/%{app_name}/LICENSE

%changelog
* Tue Mar 18 2026 Playtron [0.1.0-0]
- Initial spec file for Fedora based distribution
