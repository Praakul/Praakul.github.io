---
title: "Wayland Desktop Task Widget"
date: 2026-07-01
draft: false
projectType: "Side Projects"
tags:
- GTK
- Wayland
- Linux
- Python
- Layer Shell
description: "A persistent, transparent GTK3 desktop widget for Wayland that renders Markdown task lists directly onto the desktop wallpaper layer using gtk-layer-shell."
---
## TL;DR
A fully persistent Wayland desktop widget that renders live Markdown task files directly on the wallpaper layer — completely outside any window manager, surviving workspace switches and application overlays — with drag-to-reposition, daily auto-reset, and an in-app task editor.

## The Wayland Display Protocol Constraint

On X11, desktop widgets work by positioning frameless windows on a special "desktop" z-layer. On Wayland, this entire mechanism is absent. The compositor owns the desktop layer exclusively; regular application windows cannot reach it. The core problem is how to inject a persistent, keyboard-interactive UI element that lives below all windows but above the wallpaper itself on a modern Wayland compositor (Hyprland, Sway, River).

## Why gtk-layer-shell Instead of a Web-Based Widget

Electron-based widgets (like those in popular dotfile setups) are heavyweight and require IPC daemons. The `gtk-layer-shell` library provides direct bindings to the Wayland Layer Shell protocol (`zwlr_layer_shell_v1`), which is the compositor-level API for placing surfaces in specific stacking layers (`BACKGROUND`, `BOTTOM`, `TOP`, `OVERLAY`). Using GTK3 with Python's `gi` bindings keeps the binary small and integrates natively with the system theme engine (GTK CSS).

## Rendering on the Compositor Layer Shell

### Layer Shell Anchoring
The widget uses `GtkLayerShell` to place itself on the `BOTTOM` layer (above wallpaper, below all windows) and anchors it to screen edges. Two independent instances run simultaneously via `GLib.MainLoop`: one on the left for long-term tasks, one on the right for daily rituals.

### Markdown Live-Reload
The widget watches its Markdown source files via `GLib.timeout_add_seconds(1, self.check_file)`. When the `mtime` of `todo.md` changes (either from an in-widget toggle or external editor save), it re-parses the section headers (`# Section`) and checklist items (`- [ ]` / `- [x]`) and rebuilds the GTK widget tree in-place — zero restart required.

### True RGBA Transparency
The widget enforces a true transparent background by acquiring an RGBA `GdkVisual` from the screen and calling `set_app_paintable(True)`. The visual style is driven by a hot-reloadable `style.css`, allowing real-time theming changes.

### Daily Reset Engine
A `check_daily_reset()` method uses a logical midnight cutoff of `03:00 AM` (not calendar midnight) to avoid false resets during late-night sessions. It compares the current logical date against a persisted `.last_reset` file and automatically unchecks all daily items when a new day begins.

