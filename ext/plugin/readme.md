# Nori Exporter for Blender
 by Delio Vicini and Tizian Zeltner, based on Adrien Gruson's original exporter.

## Installation

First, you must download a fairly recent version of Blender (the plugin is tested for versions >= 4.0, including Blender 5.x). You can download blender from https://www.blender.org or using your system's package manager.

### Blender 4.2+ / 5.x (as an extension)

Use the pre-built `io_nori.zip` in this folder, or rebuild it with:

```
mkdir -p /tmp/io_nori_build/io_nori
cp io_nori.py /tmp/io_nori_build/io_nori/__init__.py
cp blender_manifest.toml /tmp/io_nori_build/io_nori/blender_manifest.toml
(cd /tmp/io_nori_build && zip -r io_nori.zip io_nori)
```

The zip must contain a single top-level `io_nori/` folder with `__init__.py` and `blender_manifest.toml` inside it — **not** the two files at the zip root (Blender will install but fail to load with "module loaded with no associated file").

Then in Blender go to "Edit -> Preferences... -> Get Extensions", click the dropdown on the top-right, and choose "Install from Disk...". Select the zip.

### Blender 4.0 – 4.1 (as a legacy add-on)

Open Blender and go to "Edit -> Preferences... -> Add-ons" and click on "Install...".
This should open a file browser in which you can navigate to the `io_nori.py` file and select it.
This will copy the exporter script to Blender's plugin directory.
After the plugin is installed, it has to be activated by clicking the checkbox next to it in the Add-ons menu.

## Usage

Once the plugin is installed, scenes can be expored by clicking "File -> Export -> Export Nori Scene..."

The plugin exports all objects in the scene as separate OBJ FIles. It then generates a Nori XML file with the scene's camera, a basic integrator and XML entries to reference all the exported meshes.

## Limitations

The plugin does not support exporting BSDFs and emitters. It will just assign a default BSDF to all shapes. It further exports each mesh as is.
This means that if you have a mesh with multiple materials in Blender, you will have to split it manually into separate submeshes before exporting (one for each material). This can be done by selecting the mesh with multiple materials, going to edit mode (tab) then selecting all vertices (A) and clicking "separate" (P) and then "by material". This will separate the mesh into meshes with one material each. After exporting, each of these meshes will have a separate entry in the scene's XML file and can therefore be assigned a different BSDF.

## More advanced features

If you need material export (diffuse / glossy / glass), area emitters, point lights, texture export, multi-material meshes, and configurable integrator / sampler / resolution / reconstruction filter settings, have a look at [Bento](https://github.com/TheCodecOfficial/Bento) — a Blender-to-Nori exporter for Blender 4.5+ with a much larger feature set.