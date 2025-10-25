import re
from json import dumps
from typing import Callable

from modules import scripts, shared

from lib_couple.attention_couple import AttentionCouple
from lib_couple.gr_version import js
from lib_couple.logging import logger
from lib_couple.mapping import (
    advanced_mapping,
    basic_mapping,
    empty_tensor,
    mask_mapping,
)
from lib_couple.tile_funcs import calculate_tiles
from lib_couple.ui import couple_ui
from lib_couple.ui_funcs import validate_mapping

try:
    from modules_forge import forge_version  # noqa
except ImportError:
    isA1111 = True
else:
    isA1111 = False


VERSION = "4.0.4"


class ForgeCouple(scripts.Script):
    forgeAttentionCouple = AttentionCouple()

    def __init__(self):
        self.is_img2img: bool
        self.couples: list | None = None
        self.neg_couples: list | None = None
        self.use_regional_neg: bool = False
        self.get_mask: Callable
        self.is_hr: bool

        self.valid: bool
        """
        Since raising error within Extensions does NOT cancel the generation,
        the only way is to forcefully interrupt during generation...
        """

        self.tile_idx: int
        self.tiles: list[str] = []

    def title(self):
        return "Forge Couple"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.is_img2img = is_img2img
        return couple_ui(
            self,
            is_img2img,
            f"{self.title()} v{VERSION}",
            (self._unpatch if isA1111 else None),
        )

    def after_component(self, component, **kwargs):
        if (elem_id := kwargs.get("elem_id", None)) is not None:
            if elem_id in ("txt2img_width", "txt2img_height"):
                component.change(None, **js('() => { ForgeCouple.preview("t2i"); }'))
            elif elem_id in ("img2img_width", "img2img_height"):
                component.change(None, **js('() => { ForgeCouple.preview("i2i"); }'))

    def setup(self, p, *args, **kwargs):
        self.is_hr = False
        if not self.is_img2img:
            return

        if calculate_tiles(self, (p, *args)) is None:
            self.invalidate(p)

        self.tile_idx = -1

    def before_process(self, p, *args, **kwargs):
        if self.tiles is None or len(self.tiles) == 0:
            return

        self.tile_idx += 1
        p.prompt = self.tiles[self.tile_idx]
        debug: bool = args[-1]

        if debug:
            print("")
            logger.info(f"[Tile Debug]\n{p.prompt}\n")

    def before_hr(self, *args, **kwargs):
        self.is_hr = True

    def _is_tile(self) -> bool:
        return self.is_img2img and len(self.tiles) > 0

    @staticmethod
    def parse_common_prompt(
        prompt: str,
        brackets: tuple[str],
        def_in_prompt: bool,
    ) -> str:
        common_prompts: dict[str, str] = {}
        op, cs = brackets

        pattern = rf"{op}([^{op}{cs}]+?):([^{op}{cs}]+?){cs}"
        matches = list(re.finditer(pattern, prompt))
        for m in matches:
            key: str = m.group(1).strip()
            val: str = m.group(2).strip()
            prompt = prompt.replace(m.group(0), val if def_in_prompt else "")
            common_prompts.update({key: val})

        pattern = rf"{op}([^{op}{cs}]+?){cs}"
        matches = list(re.finditer(pattern, prompt))
        for m in matches:
            key: str = m.group(1).strip()
            if key in common_prompts:
                prompt = prompt.replace(m.group(0), common_prompts[key])

        return prompt

    def invalidate(self, p):
        self.valid = False
        self.couples = None
        self.neg_couples = None
        self.use_regional_neg = False
        p.extra_generation_params.update({"forge_couple": "ERROR"})
        if shared.opts.fc_do_interrupt:
            shared.state.interrupt()

    def after_extra_networks_activate(
        self,
        p,
        enable: bool,
        disable_hr: bool,
        mode: str,
        separator: str,
        direction: str,
        background: str,
        background_weight: float,
        mapping: list,
        common_parser: str,
        common_debug: bool,
        def_in_prompt: bool,
        *args,
        **kwargs,
    ):
        self.couples = None
        self.neg_couples = None
        self.use_regional_neg = False
        if not enable:
            return

        if self._is_tile():
            return

        separator = separator.replace("\\n", "\n").replace("\\t", " ")
        if not separator.strip():
            separator = "\n"

        prompts: str = kwargs["prompts"][0]
        neg_prompt_source = kwargs.get("negative_prompts")
        if neg_prompt_source and neg_prompt_source[0]:
            neg_prompts: str = neg_prompt_source[0]
        else:
            neg_prompts = getattr(p, "negative_prompt", "")

        if common_parser in ("{ }", "< >"):
            prompts = self.parse_common_prompt(
                prompts,
                common_parser.split(" "),
                def_in_prompt,
            )
            neg_prompts = self.parse_common_prompt(
                neg_prompts,
                common_parser.split(" "),
                def_in_prompt,
            )
            if common_debug:
                print("")
                logger.info(f"[Common Prompts Debug]\n{prompts}\n")
                logger.info(f"[Common Negative Prompts Debug]\n{neg_prompts}\n")

        couples: list[str] = [chunk.strip() for chunk in prompts.split(separator)]
        neg_couples: list[str] = [
            chunk.strip() for chunk in neg_prompts.split(separator)
        ]

        if not any(neg_couples) and getattr(p, "negative_prompt", None):
            neg_couples = [
                chunk.strip()
                for chunk in getattr(p, "negative_prompt", "").split(separator)
            ]

        if len(neg_couples) < len(couples):
            filler = (
                neg_couples[-1]
                if neg_couples
                else getattr(p, "negative_prompt", "") or ""
            )
            filler = filler.strip()
            neg_couples.extend([filler] * (len(couples) - len(neg_couples)))

        if len(neg_couples) > len(couples):
            logger.error(
                f"Positive and negative prompt counts don't match: {len(couples)} vs {len(neg_couples)}"
            )
            self.invalidate(p)
            return

        if len(neg_couples) != len(couples):
            logger.error(
                f"Could not align negative prompts with couples: {len(neg_couples)} entries"
            )
            self.invalidate(p)
            return

        self.use_regional_neg = any(chunk.strip() for chunk in neg_couples)

        match mode:
            case "Basic":
                if len(couples) < (3 - int(background == "None")):
                    ratio = f"{len(couples)} / {3 - int(background == 'None')}"
                    logger.error(f"Not Enough Positive/Negative Lines in Prompt... [{ratio}]")
                    self.invalidate(p)
                    return

            case "Mask":
                mapping: list = self.get_mask() or mapping
                assert isinstance(mapping[0], dict)
                if not mapping:
                    logger.error("No Mapping...?")
                    self.invalidate(p)
                    return

                required: int = len(mapping) + int(background != "None")
                if len(couples) != required:
                    ratio = f"{len(couples)} / {required}"
                    logger.error(f"Number of Couples (pos/neg) and Masks mismatched... [{ratio}]")
                    self.invalidate(p)
                    return

            case "Advanced":
                assert isinstance(mapping[0], list)
                if not mapping:
                    logger.error("No Mapping...?")
                    self.invalidate(p)
                    return

                if not validate_mapping(mapping, True):
                    self.invalidate(p)
                    return

                if len(couples) != len(mapping):
                    ratio = f"{len(couples)} / {len(mapping)}"
                    logger.error(f"Number of Couples (pos/neg) and Masks mismatched... [{ratio}]")
                    self.invalidate(p)
                    return

        # ===== Infotext =====
        fc_param: dict = {}

        fc_param["forge_couple"] = True
        fc_param["forge_couple_compatibility"] = disable_hr
        fc_param["forge_couple_mode"] = mode
        fc_param["forge_couple_separator"] = separator.replace("\n", "\\n")
        fc_param["forge_couple_negative_separator"] = separator.replace("\n", "\\n")
        fc_param["forge_couple_negative_regions"] = True
        if mode == "Basic":
            fc_param["forge_couple_direction"] = direction
        if mode == "Advanced":
            fc_param["forge_couple_mapping"] = dumps(mapping)
        else:
            fc_param["forge_couple_background"] = background
            fc_param["forge_couple_background_weight"] = background_weight
        fc_param["forge_couple_common_parser"] = common_parser
        fc_param["forge_couple_def_in_prompt"] = def_in_prompt

        p.extra_generation_params.update(fc_param)
        # ===== Infotext =====

        self.couples = couples
        self.neg_couples = neg_couples
        self.valid = True

    def process_before_every_sampling(
        self,
        p,
        enable: bool,
        disable_hr: bool,
        mode: str,
        separator: str,
        direction: str,
        background: str,
        background_weight: float,
        mapping: list,
        *args,
        **kwargs,
    ):
        if (
            (not enable)
            or (self.couples is None)
            or (self.neg_couples is None)
            or (not self.valid)
        ):
            return

        if self._is_tile():
            return

        if disable_hr and self.is_hr:
            return

        if getattr(p, "_ad_inner", False):
            return

        # ===== Init =====
        if isA1111:
            unet = p.sd_model.model.diffusion_model
        else:
            unet = p.sd_model.forge_objects.unet.clone()

        WIDTH: int = p.width
        HEIGHT: int = p.height
        IS_HORIZONTAL: bool = direction == "Horizontal"
        NO_BACKGROUND: bool = background == "None"

        LINE_COUNT: int = len(self.couples)
        if len(self.neg_couples) != LINE_COUNT:
            logger.error("Positive and negative couples are out of sync before sampling.")
            self.invalidate(p)
            return

        if mode != "Advanced":
            BG_WEIGHT: float = 0.0 if NO_BACKGROUND else max(0.1, background_weight)

        if mode == "Basic":
            TILE_COUNT: int = LINE_COUNT - int(not NO_BACKGROUND)
            TILE_WEIGHT: float = 1.25 if NO_BACKGROUND else 1.0
            TILE_SIZE: int = (
                (WIDTH if IS_HORIZONTAL else HEIGHT) - 1
            ) // TILE_COUNT + 1
        # ===== Init =====

        # ===== Tiles =====
        match mode:
            case "Basic":
                fc_args = basic_mapping(
                    p.sd_model,
                    self.couples,
                    WIDTH,
                    HEIGHT,
                    LINE_COUNT,
                    IS_HORIZONTAL,
                    background,
                    TILE_SIZE,
                    TILE_WEIGHT,
                    BG_WEIGHT,
                    neg_couples=self.neg_couples,
                )

            case "Mask":
                mapping: list[dict] = self.get_mask() or mapping

                fc_args = mask_mapping(
                    p.sd_model,
                    self.couples,
                    WIDTH,
                    HEIGHT,
                    LINE_COUNT,
                    mapping,
                    background,
                    BG_WEIGHT,
                    neg_couples=self.neg_couples,
                )

            case "Advanced":
                fc_args = advanced_mapping(
                    p.sd_model,
                    self.couples,
                    WIDTH,
                    HEIGHT,
                    mapping,
                    neg_couples=self.neg_couples,
                )
        # ===== Tiles =====

        if not fc_args:
            logger.error("Forge Couple mapping produced no conditioning data.")
            self.invalidate(p)
            return

        cond_count = len([k for k in fc_args.keys() if k.startswith("cond_")])
        neg_cond_count = len([k for k in fc_args.keys() if k.startswith("neg_cond_")])
        mask_count = len([k for k in fc_args.keys() if k.startswith("mask_")])
        assert (
            cond_count == neg_cond_count == mask_count == LINE_COUNT
        ), "Mapping output mismatch between positive, negative, and masks."
        fc_args["use_regional_neg"] = self.use_regional_neg

        base_mask = empty_tensor(HEIGHT, WIDTH)
        patched_unet = self.forgeAttentionCouple.patch_unet(
            unet,
            base_mask,
            fc_args,
            isA1111=isA1111,
            width=WIDTH,
            height=HEIGHT,
        )
        if patched_unet is None:
            self.invalidate(p)
        elif not isA1111:
            p.sd_model.forge_objects.unet = patched_unet

    def postprocess(self, *args, **kwargs):
        if isA1111:
            self._unpatch()

    @classmethod
    def _unpatch(cls):
        cls.forgeAttentionCouple.unpatch(shared.sd_model.model.diffusion_model)
