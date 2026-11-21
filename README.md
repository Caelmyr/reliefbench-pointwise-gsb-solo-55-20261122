# ReliefBench

ReliefBench is a local photography workbench for studying the visible relief of small objects and reference plates. A measured physical rectangle links camera photographs, light settings, report images and stored evidence.

## Start and inspect

1. Run `./start.sh` with Docker Compose and open the address printed by the command.
2. Open either synthetic example or create a new surface with dimensions in millimetres. Add a photograph for each known directional light. Set its angle, strength, exposure and camera label.
3. Select the relevant photographs in the left pane. Choose an inspection in the right pane and run it. The right-hand controls give editable rectangles or position arrays for the mark and reference tools.
4. Save a result to keep its actual source revisions, report images and numeric tables. From Saved evidence, open the result to review it, download a PDF or check a ZIP through Verify packet. Capture sets can be imported into a new surface independently of inspection records.

The default address is `http://127.0.0.1:19958/`. The server mounts a Compose volume called `relief-library`. To stop without clearing data, run `docker compose stop`.

## Workbench tools

- Sensor acquisition: clipping and histograms, source alignment, native readout at marks, line traces, multi-photo support, local detail, physical measurements and ruler readback.
- Reference photography: neutral and color standards, fixed-pattern dark checks, radial flat-field checks, preview of per-channel flat compensation.
- Light and surface: lamp coverage and mount planning, marked response by angle, directional normals and reflectance, a withheld lamp check, slope consistency, relative height, contours and virtual lighting.
- Physical readout: topology features, material cells, line cross-sections, local texture amplitude and directional spectra. Retained data include CSV, PNG and relevant calibrated TIFF, OBJ, PLY or SVG.
- Source history: four-corner planar registrations, named setup review, snapshot-safe inspections, immutable assets, review events, PDF handouts and independently checkable evidence and capture packets.

The Slate and Ochre examples are generated. Their reference photographs include a ruler, color swatches, a flat field and a pair of dark reference exposures. Coordinates are normalized from the top-left. Azimuth is clockwise from image right with positive elevation toward the camera.

## Limits and units

PNG, JPEG, TIFF and WebP single images are accepted. The importer supports 16-bit monochrome and ordinary profiled 8-bit images. Images are limited to 9 megapixels and 18 MiB per original. A surface can hold 48 originals. One directional solve uses three to twelve known, well-separated directions. The registered grid fits within 690 pixels on its long side, and sparse height integration uses a coarser physical grid internally when needed. Inspect the source coordinate report and actual support rather than treating the working grid as newly observed information.

The directional model estimates diffuse appearance and the local normal response. Strong sheen, cast shadows, mixed exposure, missing support, or camera movement can bias it. Each connected height island has its own relative reference. Virtual lamps show local normal illumination. Open geometry has no filled holes, underside, or manufacturing allowances. Source pixels remain available in their original acquisition form.
