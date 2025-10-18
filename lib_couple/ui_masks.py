import json
import gradio as gr
import numpy as np
from PIL import Image

from .gr_version import is_gradio_4, js
from .region_shapes import BlendMode, RegionShape, ShapeType
from .shape_renderer import ShapeRenderer
from .ui_funcs import COLORS

try:
    from modules_forge.forge_canvas.canvas import ForgeCanvas
except ImportError:
    pass

BLEND_MODE_VALUES = [mode.value for mode in BlendMode]


class CoupleMaskData:
    def __init__(self, is_img2img: bool):
        self.mode: str = "i2i" if is_img2img else "t2i"
        self.masks: list[Image.Image] = []
        self.weights: list[float] = []
        self.opposite: CoupleMaskData
        self.shapes: list[RegionShape | None] = []
        self.use_shapes: bool = False
        self._shape_renderer = ShapeRenderer()
        self._shape_resolution: tuple[int, int] | None = None

        self.selected_index: int = -1

    def pull_mask(self) -> list[Image.Image]:
        """Pull the masks from the opposite tab"""
        if not (masks_data := self.opposite.get_masks()):
            self.weights = []
            self.shapes = []
            self.use_shapes = False
            return []

        first_entry = masks_data[0]
        is_shape_mapping = isinstance(first_entry, dict) and (
            "shape_type" in first_entry
            or ("shapes" in first_entry and first_entry["shapes"])
        )

        if is_shape_mapping:
            self.use_shapes = True
            self.shapes = []
            self.weights = []

            resolution = self._get_shape_canvas_size()
            rendered: list[Image.Image] = []

            for entry in masks_data:
                if not isinstance(entry, dict):
                    continue

                shape_dicts = entry.get("shapes") or [entry]
                for shape_dict in shape_dicts:
                    if not isinstance(shape_dict, dict):
                        continue

                    shape = RegionShape.from_dict(shape_dict)
                    self.shapes.append(shape)
                    self.weights.append(shape.weight)
                    mask_image = self._shape_renderer.render_shape(
                        shape, *resolution
                    ).convert("L")
                    rendered.append(mask_image)

            self._shape_resolution = resolution
            return rendered

        self.use_shapes = False
        self.shapes = []
        self.weights = [1.0 for _ in masks_data]
        return [data["mask"] for data in masks_data]

    def get_masks(self) -> list[dict]:
        """Return the current masks as well as weights"""
        if self.use_shapes:
            if not self.shapes and self.masks:
                self.shapes = []
                for index, mask in enumerate(self.masks):
                    weight = (
                        self.weights[index] if index < len(self.weights) else 1.0
                    )
                    try:
                        shape = self._convert_mask_to_shape(mask, weight, index)
                    except ValueError:
                        continue
                    self.shapes.append(shape)
                self.weights = []
                for shape in self.shapes:
                    if isinstance(shape, RegionShape):
                        self.weights.append(shape.weight)
                    else:
                        self.weights.append(1.0)
            if not self.shapes:
                return None

            return [
                shape.to_dict() for shape in self.shapes if isinstance(shape, RegionShape)
            ]

        count = len(self.masks)
        if count == 0:
            return None

        if len(self.weights) != count:
            self.weights = [1.0 for _ in range(count)]

        return [
            {"mask": self.masks[i], "weight": self.weights[i]} for i in range(count)
        ]

    def _get_shape_canvas_size(self) -> tuple[int, int]:
        if self._shape_resolution is not None:
            return self._shape_resolution
        if self.masks:
            return self.masks[0].size
        opposite = getattr(self, "opposite", None)
        if opposite and getattr(opposite, "masks", None):
            if opposite.masks:
                return opposite.masks[0].size
        return 512, 512

    def add_shape(self, shape: RegionShape):
        """Append a new RegionShape definition and keep caches in sync."""
        shape.validate()
        self.use_shapes = True
        self.shapes.append(shape)
        self.weights.append(shape.weight)

        resolution = self._get_shape_canvas_size()
        self._shape_resolution = resolution
        mask_image = self._shape_renderer.render_shape(shape, *resolution).convert("L")
        self.masks.append(mask_image)

    def _convert_mask_to_shape(
        self, mask: Image.Image, weight: float, index: int
    ) -> RegionShape:
        """Best-effort conversion from a raster mask to a RECTANGLE shape."""
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError("Cannot convert an empty mask to a RegionShape")

        x1, y1, x2, y2 = bbox
        width, height = mask.size

        parameters = {
            "x1": x1 / width,
            "y1": y1 / height,
            "x2": x2 / width,
            "y2": y2 / height,
        }

        if parameters["x1"] >= parameters["x2"]:
            if x2 < width:
                x2 = min(width, x2 + 1)
            elif x1 > 0:
                x1 = max(0, x1 - 1)
            parameters["x1"] = x1 / width
            parameters["x2"] = min(1.0, x2 / width)

        if parameters["y1"] >= parameters["y2"]:
            if y2 < height:
                y2 = min(height, y2 + 1)
            elif y1 > 0:
                y1 = max(0, y1 - 1)
            parameters["y1"] = y1 / height
            parameters["y2"] = min(1.0, y2 / height)

        shape = RegionShape(
            shape_type=ShapeType.RECTANGLE,
            parameters=parameters,
            z_order=index,
            blend_mode=BlendMode.NORMAL,
            feather=0.0,
            hardness=1.0,
            weight=float(weight),
        )
        shape.validate()
        self._shape_resolution = mask.size
        return shape

    def mask_ui(self, btn, res, mode) -> list[gr.components.Component]:
        # ===== Components ===== #
        msk_btn_empty = gr.Button("Create Empty Canvas", elem_classes="round-btn")

        gr.HTML(
            f"""
            <h2 align="center"><ins>Mask Canvas</ins></h2>
            {
                ""
                if is_gradio_4
                else '<p align="center"><b>[Important]</b> Do <b>NOT</b> upload / paste an image to here...</p>'
            }
            """
        )

        msk_canvas = (
            ForgeCanvas(scribble_color="#FFFFFF", no_upload=True)
            if is_gradio_4
            else gr.Image(
                show_label=False,
                source="upload",
                interactive=True,
                type="pil",
                tool="color-sketch",
                image_mode="RGB",
                brush_color="#ffffff",
                elem_classes="fc_msk_canvas",
            )
        )

        gr.HTML(
            """
            <div class="fc_shape_tools">
                <button type="button" class="round-btn" data-tool="RECTANGLE">Rectangle</button>
                <button type="button" class="round-btn" data-tool="ELLIPSE">Ellipse</button>
                <button type="button" class="round-btn" data-tool="POLYGON">Polygon</button>
                <button type="button" class="round-btn" data-tool="BEZIER">Bézier</button>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="fc_canvas_features">
                <label>
                    <input type="checkbox" class="fc_snap_grid">
                    Snap to Grid
                </label>
                <label>
                    Grid Size
                    <input type="number" class="fc_grid_size" min="0.01" step="0.01" value="0.10">
                </label>
                <label>
                    <input type="checkbox" class="fc_smart_guides">
                    Smart Guides
                </label>
                <label>
                    <input type="checkbox" class="fc_aspect_lock">
                    Lock Aspect
                </label>
                <label>
                    Aspect Ratio
                    <input type="text" class="fc_aspect_ratio" value="1.00">
                </label>
            </div>
            """
        )

        blend_modes_json = json.dumps(BLEND_MODE_VALUES)
        blend_options_html = "\n".join(
            f'                        <option value="{mode}">{mode}</option>'
            for mode in BLEND_MODE_VALUES
        )
        gr.HTML(
            f"""
            <div class="fc_shape_properties" data-blend-modes='{blend_modes_json}'>
                <label>
                    Blend Mode
                    <select class="fc_prop_blend">
{blend_options_html}
                    </select>
                </label>
                <label class="fc_prop_slider">
                    Edge Feather
                    <input type="range" min="0" max="1" step="0.01" value="0" class="fc_prop_feather">
                </label>
                <label class="fc_prop_slider">
                    Edge Hardness
                    <input type="range" min="0" max="1" step="0.01" value="1" class="fc_prop_hardness">
                </label>
                <label>
                    Layer Order
                    <input type="number" step="1" value="0" class="fc_prop_z">
                </label>
                <div class="fc_prop_feather_edges">
                    <label>
                        Feather Top
                        <input type="range" min="0" max="1" step="0.01" value="0" class="fc_prop_feather_edge fc_prop_feather_edge_top">
                    </label>
                    <label>
                        Feather Right
                        <input type="range" min="0" max="1" step="0.01" value="0" class="fc_prop_feather_edge fc_prop_feather_edge_right">
                    </label>
                    <label>
                        Feather Bottom
                        <input type="range" min="0" max="1" step="0.01" value="0" class="fc_prop_feather_edge fc_prop_feather_edge_bottom">
                    </label>
                    <label>
                        Feather Left
                        <input type="range" min="0" max="1" step="0.01" value="0" class="fc_prop_feather_edge fc_prop_feather_edge_left">
                    </label>
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="fc_coord_inputs">
                <label>X <input type="number" step="0.01" min="0" max="1" value="0.00"></label>
                <label>Y <input type="number" step="0.01" min="0" max="1" value="0.00"></label>
                <label>W <input type="number" step="0.01" min="0.01" max="1" value="0.10"></label>
                <label>H <input type="number" step="0.01" min="0.01" max="1" value="0.10"></label>
            </div>
            """
        )

        with gr.Row(elem_classes="fc_msk_io"):
            msk_btn_save = gr.Button(
                "Save Mask", interactive=True, elem_classes="round-btn"
            )
            msk_btn_load = gr.Button(
                "Load Mask", interactive=False, elem_classes="round-btn"
            )
            msk_btn_override = gr.Button(
                "Override Mask", interactive=False, elem_classes="round-btn"
            )

        with gr.Row(visible=False):
            operation = gr.Textbox(interactive=True, elem_classes="fc_msk_op")
            operation_btn = gr.Button("op", elem_classes="fc_msk_op_btn")

        gr.HTML('<h2 align="center"><ins>Mask Layers</ins></h2>')

        gr.HTML('<div class="fc_masks"></div>')

        shape_metadata_field = gr.Textbox(
            visible=False,
            elem_classes="fc_shape_metadata",
            show_label=False,
        )
        coords_field = gr.Textbox(
            visible=False,
            elem_classes="fc_active_shape_coords",
            show_label=False,
        )

        gr.HTML('<h2 align="center"><ins>Mask Preview</ins></h2>')

        msk_preview = gr.Image(
            show_label=False,
            image_mode="RGB",
            type="pil",
            interactive=False,
            show_download_button=False,
            elem_classes="fc_msk_preview",
        )

        msk_gallery = gr.Gallery(
            show_label=False,
            show_share_button=False,
            show_download_button=False,
            interactive=False,
            visible=False,
            elem_classes="fc_msk_gal",
        )

        msk_btn_reset = gr.Button("Reset All Masks", elem_classes="round-btn")

        msk_btn_pull = gr.Button(
            f"Pull from {'txt2img' if self.mode == 'i2i' else 'img2img'}",
            elem_classes="round-btn",
        )

        weights_field = gr.Textbox(visible=False, elem_classes="fc_msk_weights")

        dummy = None if is_gradio_4 else gr.State()

        with gr.Row(elem_classes="fc_msk_uploads"):
            upload_background = gr.Image(
                image_mode="RGBA",
                label="Upload Background",
                type="pil",
                sources="upload",
                show_download_button=False,
                interactive=True,
                height=256,
                elem_id="fc_msk_upload_bg",
            )

            upload_mask = gr.Image(
                image_mode="RGBA",
                label="Upload Mask",
                type="pil",
                sources="upload",
                show_download_button=False,
                interactive=True,
                height=256,
                elem_id="fc_msk_upload_mask",
            )

        # ===== Components ===== #

        # ===== Events ===== #
        if not is_gradio_4:
            msk_canvas.change(
                fn=None, **js(f'() => {{ ForgeCouple.hideButtons("{self.mode}"); }}')
            )
        shape_metadata_field.change(
            self._write_shape_metadata,
            inputs=[shape_metadata_field],
            outputs=[],
        )
        coords_field.change(
            self._update_shape_coords,
            inputs=[coords_field],
            outputs=[],
        )

        msk_btn_empty.click(
            fn=self._create_empty,
            inputs=[res],
            outputs=(
                [msk_canvas.background, msk_canvas.foreground]
                if is_gradio_4
                else [msk_canvas, dummy]
            ),
        )

        msk_btn_pull.click(
            self._pull_mask,
            None,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override],
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        msk_btn_save.click(
            self._write_mask,
            msk_canvas.foreground if is_gradio_4 else msk_canvas,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override],
        ).success(
            fn=self._create_empty,
            inputs=[res],
            outputs=(
                [msk_canvas.background, msk_canvas.foreground]
                if is_gradio_4
                else [msk_canvas, dummy]
            ),
        ).then(fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}'))

        msk_btn_override.click(
            self._override_mask,
            msk_canvas.foreground if is_gradio_4 else msk_canvas,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override],
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        msk_btn_load.click(
            self._load_mask, None, msk_canvas.foreground if is_gradio_4 else msk_canvas
        )

        msk_btn_reset.click(
            self._reset_masks,
            None,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override],
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        weights_field.change(self._write_weights, weights_field)

        operation_btn.click(
            self._on_operation,
            operation,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override],
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        btn.click(
            fn=self._refresh_resolution,
            inputs=[res, mode],
            outputs=(
                [msk_gallery, msk_preview, msk_canvas.background, msk_canvas.foreground]
                if is_gradio_4
                else [msk_gallery, msk_preview, msk_canvas, dummy]
            ),
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        upload_background.upload(
            fn=self._on_up_bg,
            inputs=[res, upload_background],
            outputs=[
                msk_canvas.background if is_gradio_4 else msk_canvas,
                upload_background,
            ],
        )

        upload_mask.upload(
            fn=self._on_up_mask,
            inputs=[res, upload_mask],
            outputs=[
                msk_canvas.foreground if is_gradio_4 else msk_canvas,
                upload_mask,
            ],
        )
        # ===== Events ===== #

        # ===== Pain ===== #
        [
            setattr(comp, "do_not_save_to_config", True)
            for comp in (
                msk_btn_empty,
                msk_btn_pull,
                msk_canvas,
                msk_btn_save,
                msk_btn_load,
                msk_btn_override,
                operation,
                operation_btn,
                msk_preview,
                msk_gallery,
                msk_btn_reset,
                weights_field,
                shape_metadata_field,
                coords_field,
                upload_background,
                upload_mask,
            )
        ]

        if is_gradio_4:
            msk_canvas.foreground.do_not_save_to_config = True
            msk_canvas.background.do_not_save_to_config = True
        else:
            dummy.do_not_save_to_config = True

    @staticmethod
    def _parse_resolution(resolution: str) -> tuple[int, int]:
        """Convert the resolution from width and height slider"""
        w, h = [int(v) for v in resolution.split("x")]
        while w * h > 1024 * 1024:
            w //= 2
            h //= 2

        return (w, h)

    @staticmethod
    def _create_empty(resolution: str) -> list[Image.Image, None]:
        """Generate a blank black canvas"""
        w, h = CoupleMaskData._parse_resolution(resolution)
        return [Image.new("RGB", (w, h)), None]

    @staticmethod
    def _on_up_bg(resolution: str, image: Image.Image) -> list[Image.Image, bool]:
        """Resize the uploaded image"""
        w, h = CoupleMaskData._parse_resolution(resolution)
        image = image.resize((w, h))

        matt = Image.new("RGBA", (w, h), "black")
        matt.paste(image, (0, 0), image)
        image = matt.convert("RGB")

        array = np.asarray(image, dtype=np.int16)
        array = np.clip(array - 64, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)

        return [image, gr.update(value=None)]

    @staticmethod
    def _on_up_mask(resolution: str, image: Image.Image) -> list[Image.Image, bool]:
        """Resize the uploaded image"""
        w, h = CoupleMaskData._parse_resolution(resolution)
        image = image.resize((w, h))

        if is_gradio_4:  # Only keep the pure white Mask
            image_array = np.array(image, dtype=np.uint8)
            white_mask = (image_array[..., :3] == [255, 255, 255]).all(axis=-1)
            image_array[~white_mask] = [0, 0, 0, 0]
            image = Image.fromarray(image_array)

        else:
            matt = Image.new("RGBA", (w, h))
            matt.paste(image, (0, 0), image)
            image = matt.convert("RGB")

        return [image, gr.update(value=None)]

    def _on_operation(self, op: str) -> list[list, Image.Image, bool, bool]:
        """Operations triggered from JavaScript"""
        self.selected_index = -1
        mask_update: bool = True

        # Reorder
        if "=" in op:
            from_id, to_id = [int(v) for v in op.split("=")]
            self.masks[from_id], self.masks[to_id] = (
                self.masks[to_id],
                self.masks[from_id],
            )
            if max(from_id, to_id) < len(self.weights):
                self.weights[from_id], self.weights[to_id] = (
                    self.weights[to_id],
                    self.weights[from_id],
                )
            if self.use_shapes and max(from_id, to_id) < len(self.shapes):
                self.shapes[from_id], self.shapes[to_id] = (
                    self.shapes[to_id],
                    self.shapes[from_id],
                )

        # Delete
        elif "-" in op:
            to_del = int(op.split("-")[1])
            del self.masks[to_del]
            if to_del < len(self.weights):
                del self.weights[to_del]
            if self.use_shapes and to_del < len(self.shapes):
                del self.shapes[to_del]

        # Select
        else:
            self.selected_index = int(op.strip())
            mask_update = False

        return [
            self.masks if mask_update else gr.skip(),
            self._generate_preview() if mask_update else gr.skip(),
            gr.update(interactive=(self.selected_index >= 0)),
            gr.update(interactive=(self.selected_index >= 0)),
        ]

    def _generate_preview(self) -> Image.Image:
        """Create a preview based on cached masks"""
        if self.use_shapes:
            if not self.shapes:
                return None

            res = self._get_shape_canvas_size()
            self._shape_resolution = res
            bg = Image.new("RGBA", res, "black")

            for i, shape in enumerate(self.shapes):
                if not isinstance(shape, RegionShape):
                    continue
                mask_image = self._shape_renderer.render_shape(shape, *res)
                alpha_array = (np.asarray(mask_image, dtype=np.uint8) * 144) // 255
                alpha = Image.fromarray(alpha_array.astype(np.uint8))
                color = Image.new("RGB", res, COLORS[i % 7])
                rgba = Image.merge("RGBA", [*color.split(), alpha])
                bg.paste(rgba, (0, 0), rgba)

            return bg

        if not self.masks:
            return None

        res: tuple[int, int] = self.masks[0].size
        bg = Image.new("RGBA", res, "black")

        for i, mask in enumerate(self.masks):
            color = Image.new("RGB", res, COLORS[i % 7])
            alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 144)
            rgba = Image.merge("RGBA", [*color.split(), alpha.convert("L")])
            bg.paste(rgba, (0, 0), rgba)

        return bg

    def _refresh_resolution(
        self, resolution: str, mode: str
    ) -> list[list, Image.Image, Image.Image, None]:
        """Refresh when width or height is changed"""

        if mode != "Mask":
            return [gr.skip(), gr.skip(), None, None]

        (canvas, _) = self._create_empty(resolution)

        w, h = self._parse_resolution(resolution)

        if self.use_shapes and self.shapes:
            self._shape_resolution = (w, h)
            self.masks = [
                self._shape_renderer.render_shape(shape, w, h).convert("L")
                for shape in self.shapes
                if isinstance(shape, RegionShape)
            ]
        else:
            self.masks = [mask.resize((w, h)) for mask in self.masks]

        preview = self._generate_preview()

        return [self.masks, preview, canvas, None]

    def _reset_masks(self) -> list[list, Image.Image, bool, bool]:
        """Clear everything"""
        self.masks.clear()
        self.weights.clear()
        self.shapes.clear()
        self.use_shapes = False
        self._shape_resolution = None
        preview = self._generate_preview()

        return [
            self.masks,
            preview,
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    def _load_mask(self) -> Image.Image:
        """Load a cached mask to canvas based on index"""
        return self.masks[self.selected_index]

    def _override_mask(
        self, img: None | Image.Image
    ) -> list[list, Image.Image, bool, bool]:
        """Override a cached mask based on index"""
        if img is None:
            self.selected_index = -1
            return [
                self.masks,
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

        assert isinstance(img, Image.Image)

        array = np.asarray(img.convert("L"), dtype=np.uint8)
        mask = np.where(array == 255, 255, 0)
        img = Image.fromarray(mask.astype(np.uint8))

        if not bool(img.getbbox()):
            self.selected_index = -1
            return [
                self.masks,
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

        index = self.selected_index
        processed = img.convert("L" if self.use_shapes else "1")
        self.masks[index] = processed

        if index >= len(self.weights):
            self.weights.extend([1.0] * (index - len(self.weights) + 1))

        if self.use_shapes:
            weight = self.weights[index]
            try:
                shape = self._convert_mask_to_shape(processed, weight, index)
            except ValueError:
                pass
            else:
                if index < len(self.shapes):
                    self.shapes[index] = shape
                else:
                    self.shapes.append(shape)

        self.selected_index = -1

        preview = self._generate_preview()
        return [
            self.masks,
            preview,
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    def _write_mask(
        self, img: None | Image.Image
    ) -> list[list, Image.Image, bool, bool]:
        """Save a new mask"""
        if img is None:
            return [
                self.masks,
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

        assert isinstance(img, Image.Image)

        array = np.asarray(img.convert("L"), dtype=np.uint8)
        mask = np.where(array == 255, 255, 0)
        img = Image.fromarray(mask.astype(np.uint8))

        if not bool(img.getbbox()):
            return [
                self.masks,
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

        processed = img.convert("L" if self.use_shapes else "1")
        self.masks.append(processed)

        if self.use_shapes:
            index = len(self.masks) - 1
            weight = self.weights[index] if index < len(self.weights) else 1.0
            if index >= len(self.weights):
                self.weights.append(weight)
            try:
                shape = self._convert_mask_to_shape(processed, weight, index)
            except ValueError:
                pass
            else:
                self.shapes.append(shape)

        preview = self._generate_preview()
        return [
            self.masks,
            preview,
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    def _pull_mask(self) -> list[list, Image.Image, bool, bool]:
        """Pull masks from opposite tab"""

        self.masks: list[Image.Image] = self.pull_mask()

        preview = self._generate_preview()
        return [
            self.masks,
            preview,
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    def _write_shape_metadata(self, metadata: str):
        """Receive JSON shape metadata from the front-end."""
        if metadata is None:
            return

        metadata = metadata.strip()
        if not metadata:
            self.shapes = []
            self.use_shapes = False
            return

        try:
            entries = json.loads(metadata)
        except json.JSONDecodeError:
            return

        if not isinstance(entries, list):
            return

        parsed: list[RegionShape | None] = []
        for entry in entries:
            if not entry:
                parsed.append(None)
                continue
            try:
                shape = RegionShape.from_dict(entry)
            except (ValueError, KeyError):
                parsed.append(None)
            else:
                parsed.append(shape)

        self.shapes = parsed
        self.use_shapes = any(isinstance(shape, RegionShape) for shape in parsed)

        if self.use_shapes:
            new_weights: list[float] = []
            for index, shape in enumerate(parsed):
                if isinstance(shape, RegionShape):
                    new_weights.append(float(shape.weight))
                else:
                    existing = self.weights[index] if index < len(self.weights) else 1.0
                    new_weights.append(existing)
            self.weights = new_weights

    def _update_shape_coords(self, coords: str):
        """Update the currently selected shape based on coordinate input."""
        if not coords or self.selected_index < 0 or self.selected_index >= len(self.shapes):
            return

        shape = self.shapes[self.selected_index]
        if not isinstance(shape, RegionShape):
            return

        try:
            data = json.loads(coords)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        x = self._clamp01(float(data.get("x", 0.0)))
        y = self._clamp01(float(data.get("y", 0.0)))
        width = self._clamp01(float(data.get("w", 0.0)))
        height = self._clamp01(float(data.get("h", 0.0)))

        new_bounds = {
            "x1": x,
            "y1": y,
            "x2": self._clamp01(x + width),
            "y2": self._clamp01(y + height),
        }

        if shape.shape_type is ShapeType.RECTANGLE:
            shape.parameters["x1"] = new_bounds["x1"]
            shape.parameters["y1"] = new_bounds["y1"]
            shape.parameters["x2"] = new_bounds["x2"]
            shape.parameters["y2"] = new_bounds["y2"]
        elif shape.shape_type is ShapeType.ELLIPSE:
            shape.parameters["cx"] = (new_bounds["x1"] + new_bounds["x2"]) / 2
            shape.parameters["cy"] = (new_bounds["y1"] + new_bounds["y2"]) / 2
            shape.parameters["rx"] = self._clamp01(max((new_bounds["x2"] - new_bounds["x1"]) / 2, 0))
            shape.parameters["ry"] = self._clamp01(max((new_bounds["y2"] - new_bounds["y1"]) / 2, 0))
        elif shape.shape_type in (ShapeType.POLYGON, ShapeType.BEZIER):
            original_bounds = self._shape_bounds(shape)
            if original_bounds is None:
                return

            if shape.shape_type is ShapeType.POLYGON:
                points = shape.parameters.get("points", [])
                shape.parameters["points"] = self._transform_points(points, original_bounds, new_bounds)
            else:
                control_points = shape.parameters.get("control_points", [])
                shape.parameters["control_points"] = self._transform_points(
                    control_points,
                    original_bounds,
                    new_bounds,
                )
        else:
            return

        try:
            shape.validate()
        except ValueError:
            return

        self.shapes[self.selected_index] = shape
        if self.selected_index < len(self.weights):
            self.weights[self.selected_index] = float(shape.weight)

    def _write_weights(self, weights: str):
        """Cache the mask weights"""
        if not weights.strip():
            self.weights = []
        else:
            self.weights = [float(v) for v in weights.split(",")]

        if self.use_shapes:
            weight_index = 0
            for shape in self.shapes:
                if not isinstance(shape, RegionShape):
                    continue
                if weight_index < len(self.weights):
                    shape.weight = float(self.weights[weight_index])
                weight_index += 1

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(min(max(value, 0.0), 1.0))

    @staticmethod
    def _transform_points(points, original: dict[str, float], new: dict[str, float]):
        if not isinstance(points, list):
            return []
        old_width = max(original["x2"] - original["x1"], 1e-6)
        old_height = max(original["y2"] - original["y1"], 1e-6)
        new_width = max(new["x2"] - new["x1"], 1e-6)
        new_height = max(new["y2"] - new["y1"], 1e-6)

        transformed = []
        for entry in points:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            px, py = float(entry[0]), float(entry[1])
            rel_x = 0.5 if old_width <= 1e-6 else (px - original["x1"]) / old_width
            rel_y = 0.5 if old_height <= 1e-6 else (py - original["y1"]) / old_height
            nx = new["x1"] + rel_x * new_width
            ny = new["y1"] + rel_y * new_height
            transformed.append([
                CoupleMaskData._clamp01(nx),
                CoupleMaskData._clamp01(ny),
            ])
        return transformed

    @staticmethod
    def _shape_bounds(shape: RegionShape) -> dict[str, float] | None:
        params = shape.parameters
        if shape.shape_type is ShapeType.RECTANGLE:
            return {
                "x1": CoupleMaskData._clamp01(float(params.get("x1", 0.0))),
                "y1": CoupleMaskData._clamp01(float(params.get("y1", 0.0))),
                "x2": CoupleMaskData._clamp01(float(params.get("x2", 0.0))),
                "y2": CoupleMaskData._clamp01(float(params.get("y2", 0.0))),
            }
        if shape.shape_type is ShapeType.ELLIPSE:
            cx = CoupleMaskData._clamp01(float(params.get("cx", 0.0)))
            cy = CoupleMaskData._clamp01(float(params.get("cy", 0.0)))
            rx = float(params.get("rx", 0.0))
            ry = float(params.get("ry", 0.0))
            return {
                "x1": CoupleMaskData._clamp01(cx - rx),
                "x2": CoupleMaskData._clamp01(cx + rx),
                "y1": CoupleMaskData._clamp01(cy - ry),
                "y2": CoupleMaskData._clamp01(cy + ry),
            }
        if shape.shape_type is ShapeType.POLYGON:
            points = params.get("points", [])
            xs = [CoupleMaskData._clamp01(float(pt[0])) for pt in points if isinstance(pt, (list, tuple)) and len(pt) == 2]
            ys = [CoupleMaskData._clamp01(float(pt[1])) for pt in points if isinstance(pt, (list, tuple)) and len(pt) == 2]
            if not xs or not ys:
                return None
            return {
                "x1": min(xs),
                "x2": max(xs),
                "y1": min(ys),
                "y2": max(ys),
            }
        if shape.shape_type is ShapeType.BEZIER:
            control_points = params.get("control_points", [])
            return CoupleMaskData._bezier_bounds(control_points)
        return None

    @staticmethod
    def _bezier_bounds(control_points, samples: int = 32) -> dict[str, float] | None:
        if not isinstance(control_points, list) or len(control_points) < 4:
            return None

        coords: list[tuple[float, float]] = []
        for entry in control_points:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            coords.append((float(entry[0]), float(entry[1])))

        if len(coords) < 4:
            return None

        segment_count = (len(coords) - 1) // 3
        if segment_count <= 0:
            return None

        min_x, max_x = 1.0, 0.0
        min_y, max_y = 1.0, 0.0

        for idx in range(segment_count):
            p0, p1, p2, p3 = coords[idx * 3 : idx * 3 + 4]
            for j, t in enumerate(np.linspace(0.0, 1.0, samples, endpoint=True)):
                if idx > 0 and j == 0:
                    continue
                omt = 1.0 - t
                omt2 = omt * omt
                t2 = t * t
                x = (
                    omt2 * omt * p0[0]
                    + 3 * omt2 * t * p1[0]
                    + 3 * omt * t2 * p2[0]
                    + t2 * t * p3[0]
                )
                y = (
                    omt2 * omt * p0[1]
                    + 3 * omt2 * t * p1[1]
                    + 3 * omt * t2 * p2[1]
                    + t2 * t * p3[1]
                )
                clamped_x = CoupleMaskData._clamp01(x)
                clamped_y = CoupleMaskData._clamp01(y)
                min_x = min(min_x, clamped_x)
                max_x = max(max_x, clamped_x)
                min_y = min(min_y, clamped_y)
                max_y = max(max_y, clamped_y)

        return {
            "x1": min_x,
            "x2": max_x,
            "y1": min_y,
            "y2": max_y,
        }
