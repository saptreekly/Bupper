{pkgs}: {
  deps = [
    pkgs.vulkan-loader
    pkgs.libxkbcommon
    pkgs.wayland
    pkgs.systemd
    pkgs.alsa-lib
    pkgs.libudev
    pkgs.glibcLocales
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
  ];
}
