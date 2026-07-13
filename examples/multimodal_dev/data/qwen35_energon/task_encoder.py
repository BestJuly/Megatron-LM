# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL descriptor-aware Energon TaskEncoder.

The encoder packs text and image metadata before materializing the selected
pack. Images may come from JSON-only lazy descriptors or prepared ``.jpgs``
byte payloads; both follow the same descriptor materialization contract.
"""

from __future__ import annotations

import io
import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image

from examples.multimodal_dev.data.energon_vision_balance import (
    assign_images_lpt,
    image_costs_from_grid,
    vision_rows_from_grid,
)
from examples.multimodal_dev.mdp_image_materialize import (
    encode_image_descriptors,
    materialize_descriptor,
)
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_END_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from megatron.energon import SampleDecoder, TaskEncoder, stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from .raw_jpgs import _load_image_bytes_payload

_MIMO_PACK_PAD_MULTIPLE = 64
_RAW_JPGS_MATERIALIZER = "examples.multimodal_dev.data.qwen35_energon.raw_jpgs"
_GENERATION_BLOCK_RE = re.compile(r"{%-?\s*generation\b")


@stateless
def _cook_qwen35(sample: dict) -> dict:
    out = dict(**basic_sample_keys(sample), json=sample["json"])
    if "jpgs" in sample:
        out["jpgs"] = sample["jpgs"]
    return out


class Qwen35EnergonTaskEncoder(TaskEncoder):
    """Metadata-first Qwen3.5-VL TaskEncoder for Energon CrudeWebdataset."""

    cookers = [Cooker(_cook_qwen35, has_subflavors={"crude_type": "qwen35"})]
    decoder = SampleDecoder(image_decode="pil")

    def __init__(
        self,
        *,
        tokenizer,
        seq_length: int,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        vision_end_token_id: int = QWEN35_VL_VISION_END_TOKEN_ID,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        image_max_pixels: int = 1280 * 32 * 32,
        image_min_pixels: int = 256 * 32 * 32,
        cp_size: int = 1,
        mdp_loader_prepartition: bool = False,
        mdp_loader_prepartition_rank: int = 0,
        mdp_loader_prepartition_world: int = 1,
        mdp_loader_prepartition_encoder_stage: bool = True,
        mdp_loader_prepartition_materialize: bool = True,
        mdp_lpt_hidden_size: int = 1152,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = int(seq_length)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.spatial_merge_size = int(spatial_merge_size)
        self.image_max_pixels = int(image_max_pixels)
        self.image_min_pixels = int(image_min_pixels)
        self.cp_size = int(cp_size)
        self.align = max(2 * self.cp_size, _MIMO_PACK_PAD_MULTIPLE)
        self.mdp_loader_prepartition = bool(mdp_loader_prepartition)
        self.mdp_loader_prepartition_rank = int(mdp_loader_prepartition_rank)
        self.mdp_loader_prepartition_world = max(
            1, int(mdp_loader_prepartition_world)
        )
        self.mdp_loader_prepartition_encoder_stage = bool(
            mdp_loader_prepartition_encoder_stage
        )
        self.mdp_loader_prepartition_materialize = bool(
            mdp_loader_prepartition_materialize
        )
        self.mdp_lpt_hidden_size = int(mdp_lpt_hidden_size)
        self._pixel_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size

    @staticmethod
    def _load_payload(value):
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, list):
            return {"conversation": value}
        if isinstance(value, dict):
            return value
        raise ValueError(f"unsupported qwen35 Energon payload type: {type(value).__name__}")

    @staticmethod
    def _load_image_bytes(value) -> List[bytes]:
        if value is None:
            return []
        if isinstance(value, (bytes, bytearray)):
            return _load_image_bytes_payload(bytes(value))
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError(
                f"unsupported qwen35 Energon jpgs payload type: {type(value).__name__}"
            )
        return [bytes(item) for item in value]

    def _descriptor_from_image(self, image):
        if image is None:
            return None
        if isinstance(image, (bytes, bytearray)):
            image_bytes = bytes(image)
            with Image.open(io.BytesIO(image_bytes)) as im:
                width, height = im.size
        else:
            width, height = image.size
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            image_bytes = buf.getvalue()
        return {
            "kind": "image_bytes",
            "materializer": "examples.multimodal_dev.data.qwen35_energon.raw_jpgs",
            "image_bytes": image_bytes,
            "width": int(width),
            "height": int(height),
            "min_pixels": int(self.image_min_pixels),
            "max_pixels": int(self.image_max_pixels),
            "pixel_dim": int(self._pixel_dim),
            "patch_size": int(self.patch_size),
            "spatial_merge_size": int(self.spatial_merge_size),
            "temporal_patch_size": int(self.temporal_patch_size),
        }

    def _load_vqa_sample(self, sample):
        if not all(hasattr(sample, name) for name in ("context", "answers", "image")):
            return None
        answer = ""
        answers = getattr(sample, "answers", None) or []
        if isinstance(answers, str):
            answer = answers
        elif answers:
            answer = str(answers[0] or "")
        payload = {
            "conversation": [
                {"role": "user", "content": str(getattr(sample, "context", "") or "")},
                {"role": "assistant", "content": answer},
            ]
        }
        descriptor = self._descriptor_from_image(getattr(sample, "image", None))
        descriptors = [descriptor] if descriptor is not None else []
        payload["image_descriptors"] = descriptors
        return payload

    @staticmethod
    def _normalize_role(role: Any) -> str:
        role = str(role or "user").lower()
        if role in ("human", "user"):
            return "user"
        if role in ("gpt", "assistant", "model"):
            return "assistant"
        if role == "system":
            return "system"
        return "user"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                    elif part.get("type") == "image":
                        parts.append("<image>")
                    elif part.get("type") == "video":
                        parts.append("<video>")
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)

    @classmethod
    def _parse_text_turns(cls, text: str) -> List[Dict[str, str]]:
        markers = list(
            re.finditer(r"(?im)^(system|user|human|assistant|gpt|model)\s*:\s*", text or "")
        )
        if not markers:
            return []
        turns = []
        for idx, marker in enumerate(markers):
            start = marker.end()
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
            content = text[start:end].strip()
            if content:
                turns.append({"role": cls._normalize_role(marker.group(1)), "content": content})
        return turns

    @classmethod
    def _conversation_turns(cls, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        text = payload.get("text")
        if text is not None:
            text = str(text)
            turns = cls._parse_text_turns(text)
            if turns:
                return turns
            return [{"role": "user", "content": text}]
        conversation = (
            payload.get("conversation")
            or payload.get("conversations")
            or payload.get("messages")
            or []
        )
        turns = []
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))
            turns.append(
                {"role": cls._normalize_role(role), "content": cls._content_to_text(content)}
            )
        return turns

    @staticmethod
    def _normalize_token_ids(ids) -> List[int]:
        if ids is None:
            return []
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().reshape(-1).tolist()
        elif hasattr(ids, "ids"):
            ids = ids.ids
        return [int(value) for value in ids]

    def _text_to_tokens(self, text: str, length: Optional[int] = None) -> torch.Tensor:
        token_ids = self._normalize_token_ids(
            self.tokenizer.encode(text or "", add_special_tokens=False)
        )
        if length is not None:
            token_ids = token_ids[: max(0, int(length))]
        return torch.tensor(token_ids, dtype=torch.long)

    def _loss_mask_excluded_token_ids(self) -> Tuple[int, ...]:
        token_ids = {
            0,
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        }
        for token_id in getattr(self.tokenizer, "all_special_ids", []) or []:
            if token_id is not None:
                token_ids.add(int(token_id))
        return tuple(sorted(token_ids))

    def _zero_loss_mask_for_special_tokens(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        for token_id in self._loss_mask_excluded_token_ids():
            loss_mask[input_ids == int(token_id)] = 0.0
        return loss_mask

    def _split_text_with_image_parts(
        self, text: str, *, start_image_idx: int, max_images: int
    ) -> Tuple[List[Dict[str, str]], int]:
        parts = []
        image_idx = int(start_image_idx)
        cur = 0
        for match in re.finditer(r"<image>", text or ""):
            if match.start() > cur:
                parts.append({"type": "text", "text": text[cur : match.start()]})
            if image_idx < max_images:
                parts.append({"type": "image", "image": str(image_idx)})
                image_idx += 1
            cur = match.end()
        if cur < len(text or ""):
            parts.append({"type": "text", "text": text[cur:]})
        return parts, image_idx

    def _chat_conversation_with_images(
        self, turns: Sequence[Dict[str, str]], num_images: int
    ) -> List[Dict[str, Any]]:
        out = []
        image_idx = 0
        first_user_idx = None
        for turn in turns:
            role = self._normalize_role(turn.get("role", "user"))
            text = self._content_to_text(turn.get("content", ""))
            if first_user_idx is None and role == "user":
                first_user_idx = len(out)
            parts, image_idx = self._split_text_with_image_parts(
                text, start_image_idx=image_idx, max_images=num_images
            )
            has_media = any(part.get("type") == "image" for part in parts)
            content = parts if has_media else text
            out.append({"role": role, "content": content})

        if image_idx < num_images:
            media = [{"type": "image", "image": str(i)} for i in range(image_idx, num_images)]
            target_idx = first_user_idx if first_user_idx is not None else 0
            if not out:
                out.append({"role": "user", "content": media})
            else:
                content = out[target_idx]["content"]
                if isinstance(content, list):
                    out[target_idx]["content"] = media + content
                elif content:
                    out[target_idx]["content"] = media + [{"type": "text", "text": str(content)}]
                else:
                    out[target_idx]["content"] = media

        return out

    def _chat_template_to_tokens(
        self, turns: Sequence[Dict[str, str]], image_token_counts: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tokenizer = self.tokenizer

        if not hasattr(tokenizer, "apply_chat_template"):
            supervise_all_text = not any(
                self._normalize_role(turn.get("role", "user")) == "assistant" for turn in turns
            )
            parts = [self._image_block(n_tok) for n_tok in image_token_counts]
            masks = [torch.zeros(int(part.numel()), dtype=torch.float32) for part in parts]
            for turn in turns:
                role = self._normalize_role(turn.get("role", "user"))
                content = self._content_to_text(turn.get("content", ""))
                prefix = self._text_to_tokens(f"{role}: ")
                content_tokens = self._text_to_tokens(content)
                suffix = self._text_to_tokens("\n")
                parts.extend([prefix, content_tokens, suffix])
                masks.extend(
                    [
                        torch.zeros(int(prefix.numel()), dtype=torch.float32),
                        torch.full(
                            (int(content_tokens.numel()),),
                            1.0 if role == "assistant" or supervise_all_text else 0.0,
                            dtype=torch.float32,
                        ),
                        torch.zeros(int(suffix.numel()), dtype=torch.float32),
                    ]
                )
            input_ids = torch.cat([part for part in parts if int(part.numel()) > 0])
            assistant_mask = torch.cat([part for part in masks if int(part.numel()) > 0])
            input_ids = input_ids[: self.seq_length]
            assistant_mask = assistant_mask[: self.seq_length]
            assistant_mask = self._zero_loss_mask_for_special_tokens(input_ids, assistant_mask)
            self._validate_assistant_supervision(turns, assistant_mask)
            return (input_ids, assistant_mask, len(image_token_counts))

        conversation = self._chat_conversation_with_images(turns, len(image_token_counts))
        token_ids, assistant_masks = self._apply_chat_template_with_assistant_mask(
            tokenizer, conversation
        )
        out: List[int] = []
        mask_out: List[float] = []
        image_idx = 0
        i = 0
        while i < len(token_ids):
            token_id = int(token_ids[i])
            token_mask = float(assistant_masks[i])
            if token_id == self.image_token_id and image_idx < len(image_token_counts):
                tail = []
                tail_mask = []
                skip = 1
                if i + 1 < len(token_ids) and int(token_ids[i + 1]) == self.vision_end_token_id:
                    tail.append(self.vision_end_token_id)
                    tail_mask.append(0.0)
                    skip = 2
                n_tokens = int(image_token_counts[image_idx])
                if len(out) + n_tokens + len(tail) > self.seq_length:
                    if out and out[-1] == self.vision_start_token_id:
                        out.pop()
                        mask_out.pop()
                    break
                out.extend([self.image_token_id] * n_tokens)
                mask_out.extend([0.0] * n_tokens)
                out.extend(tail)
                mask_out.extend(tail_mask)
                image_idx += 1
                i += skip
                continue
            if len(out) + 1 > self.seq_length:
                break
            out.append(token_id)
            if token_id in (
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            ):
                token_mask = 0.0
            mask_out.append(token_mask)
            i += 1

        input_ids = torch.tensor(out, dtype=torch.long)
        assistant_mask = torch.tensor(mask_out, dtype=torch.float32)
        assistant_mask = self._zero_loss_mask_for_special_tokens(input_ids, assistant_mask)
        self._validate_assistant_supervision(turns, assistant_mask)
        return input_ids, assistant_mask, image_idx

    def _apply_chat_template_with_assistant_mask(
        self, tokenizer, conversation: Sequence[Dict[str, Any]]
    ) -> Tuple[List[int], List[float]]:
        has_assistant = any(
            self._normalize_role(turn.get("role", "user")) == "assistant" for turn in conversation
        )
        if has_assistant:
            generation_result = self._generation_block_assistant_mask(tokenizer, conversation)
            if generation_result is not None and any(generation_result[1]):
                return generation_result

        ids = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False
        )
        token_ids = self._normalize_token_ids(ids)
        target_roles = {"assistant"} if has_assistant else {"user", "assistant"}
        mask = self._target_role_mask_from_chatml_boundaries(
            tokenizer, conversation, token_ids, target_roles
        )
        if mask is None:
            raise ValueError(
                "Unable to build assistant loss mask from the tokenizer chat template. "
                "The template must expose generation masks or ChatML role boundaries."
            )
        return token_ids, mask

    def _generation_block_assistant_mask(
        self, tokenizer, conversation: Sequence[Dict[str, Any]]
    ) -> Optional[Tuple[List[int], List[float]]]:
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or _GENERATION_BLOCK_RE.search(template) is None:
            return None
        try:
            rendered = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if not isinstance(rendered, Mapping) or "assistant_masks" not in rendered:
            return None
        token_ids = self._normalize_token_ids(rendered.get("input_ids"))
        mask = [float(value) for value in self._normalize_token_ids(rendered["assistant_masks"])]
        if len(token_ids) != len(mask):
            return None
        return token_ids, mask

    def _target_role_mask_from_chatml_boundaries(
        self,
        tokenizer,
        conversation: Sequence[Dict[str, Any]],
        token_ids: Sequence[int],
        target_roles: set,
    ) -> Optional[List[float]]:
        end_ids = self._normalize_token_ids(
            tokenizer.encode("<|im_end|>", add_special_tokens=False)
        )
        if not end_ids:
            return None

        mask = [0.0] * len(token_ids)
        cursor = 0
        for turn in conversation:
            role = self._normalize_role(turn.get("role", "user"))
            start_ids = self._normalize_token_ids(
                tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
            )
            start = self._find_subsequence(token_ids, start_ids, cursor)
            if start < 0:
                return None
            content_start = start + len(start_ids)
            content_end = self._find_subsequence(token_ids, end_ids, content_start)
            if content_end < 0:
                return None
            if role in target_roles:
                mask[content_start:content_end] = [1.0] * (content_end - content_start)
            cursor = content_end + len(end_ids)
        return mask

    def _validate_assistant_supervision(
        self, turns: Sequence[Dict[str, Any]], assistant_mask: torch.Tensor
    ) -> None:
        has_nonempty_assistant = any(
            self._normalize_role(turn.get("role", "user")) == "assistant"
            and self._content_to_text(turn.get("content", "")).strip()
            for turn in turns
        )
        if has_nonempty_assistant and int(torch.count_nonzero(assistant_mask).item()) == 0:
            raise ValueError(
                "Chat template produced no supervised tokens for a non-empty assistant turn."
            )

    @staticmethod
    def _find_subsequence(values: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
        if not pattern:
            return -1
        last = len(values) - len(pattern)
        for idx in range(max(0, int(start)), last + 1):
            if list(values[idx : idx + len(pattern)]) == list(pattern):
                return idx
        return -1

    def _resize_hw(self, width: int, height: int) -> Tuple[int, int]:
        unit = self.patch_size * self.spatial_merge_size
        h = int(height)
        w = int(width)
        min_pixels = int(self.image_min_pixels)
        max_pixels = int(self.image_max_pixels)
        if max_pixels > 0 and h * w > max_pixels:
            scale = math.sqrt(max_pixels / (h * w))
            h = math.floor(h * scale / unit) * unit
            w = math.floor(w * scale / unit) * unit
        elif min_pixels > 0 and h * w < min_pixels:
            scale = math.sqrt(min_pixels / (h * w))
            h = math.ceil(h * scale / unit) * unit
            w = math.ceil(w * scale / unit) * unit
        else:
            h = round(h / unit) * unit
            w = round(w / unit) * unit
        return max(unit, h), max(unit, w)

    def _grid_from_descriptor(self, descriptor: Dict[str, Any]):
        grid = descriptor.get("grid_thw")
        if (
            grid is None
            and "_raw_image_bytes" in descriptor
            and ("width" not in descriptor or "height" not in descriptor)
        ):
            with Image.open(io.BytesIO(bytes(descriptor["_raw_image_bytes"]))) as im:
                descriptor["width"] = int(im.size[0])
                descriptor["height"] = int(im.size[1])
        has_resize_budget = self.image_max_pixels > 0 or self.image_min_pixels > 0
        if has_resize_budget and "width" in descriptor and "height" in descriptor:
            t = int(grid[0]) if grid is not None else 1
            h, w = self._resize_hw(int(descriptor["width"]), int(descriptor["height"]))
            return t, h // self.patch_size, w // self.patch_size
        if grid is not None:
            return tuple(int(x) for x in grid)
        if "width" not in descriptor or "height" not in descriptor:
            raise ValueError("qwen35 Energon descriptor must contain grid_thw or width/height")
        h, w = self._resize_hw(int(descriptor["width"]), int(descriptor["height"]))
        return 1, h // self.patch_size, w // self.patch_size

    def _image_block(self, n_tokens: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.tensor([self.vision_start_token_id], dtype=torch.long),
                torch.full((int(n_tokens),), self.image_token_id, dtype=torch.long),
                torch.tensor([self.vision_end_token_id], dtype=torch.long),
            ]
        )

    def _shifted_labels_and_loss_mask(
        self,
        input_ids: torch.Tensor,
        assistant_mask: torch.Tensor,
        segment_lens: Sequence[int],
        content_lens: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if int(assistant_mask.numel()) > 1:
            loss_mask[:-1] = assistant_mask[1:].to(dtype=torch.float32)
        for token_id in self._loss_mask_excluded_token_ids():
            loss_mask[labels == int(token_id)] = 0.0
        input_len = int(input_ids.numel())
        offset = 0
        for seg_len, content_len in zip(segment_lens, content_lens):
            seg_len = int(seg_len)
            content_len = int(content_len)
            if content_len > 0:
                loss_mask[offset + content_len - 1 : offset + seg_len] = 0.0
            else:
                loss_mask[offset : offset + seg_len] = 0.0
            offset += seg_len
        if offset < input_len:
            loss_mask[offset:] = 0.0
        loss_mask[-1] = 0.0
        labels[loss_mask == 0.0] = -100
        return labels, loss_mask

    def _mimo_padded_len(self, length: int) -> int:
        length = int(length)
        if length <= 0:
            return 0
        align = max(1, int(self.align))
        return int(math.ceil(length / align) * align)

    def _pad_doc(self, doc: dict, max_length: Optional[int] = None) -> Optional[dict]:
        content_len = int(doc.get("content_len", doc["real_len"]))
        padded_len = self._mimo_padded_len(content_len)
        if max_length is not None and padded_len > int(max_length):
            return None
        input_ids = doc["input_ids"].reshape(-1)[:content_len].clone()
        assistant_mask = doc["_assistant_loss_mask"].reshape(-1)[:content_len].clone()
        if padded_len > content_len:
            input_ids = torch.cat(
                [input_ids, torch.zeros(padded_len - content_len, dtype=torch.long)]
            )
            assistant_mask = torch.cat(
                [assistant_mask, torch.zeros(padded_len - content_len, dtype=torch.float32)]
            )
        out = dict(doc)
        out["input_ids"] = input_ids
        out["_assistant_loss_mask"] = assistant_mask
        out["content_len"] = content_len
        out["real_len"] = padded_len
        return out

    def _position_ids_for_doc(
        self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        return position_ids.squeeze(1)

    @stateless
    def preencode_sample(self, sample):
        raw_image_bytes = []
        vqa_payload = self._load_vqa_sample(sample)
        if vqa_payload is not None:
            payload = vqa_payload
        elif isinstance(sample, dict):
            payload = self._load_payload(sample.get("json", sample.get("txt", sample)))
            jpg = sample.get("jpg")
            if jpg is not None:
                descriptor = self._descriptor_from_image(jpg)
                if descriptor is not None:
                    payload = dict(payload)
                    payload["image_descriptors"] = [descriptor] + list(
                        payload.get("image_descriptors") or payload.get("images") or []
                    )
            else:
                raw_image_bytes = self._load_image_bytes(sample.get("jpgs"))
        else:
            payload = self._load_payload(sample)
        descriptors = [
            dict(desc) for desc in (payload.get("image_descriptors") or payload.get("images") or [])
        ]
        if not descriptors and raw_image_bytes:
            descriptors = [{} for _ in raw_image_bytes]
        turns = self._conversation_turns(payload)

        image_grids = []
        image_patch_counts = []
        image_token_counts = []
        image_block_cost = 0
        for descriptor_idx, descriptor in enumerate(descriptors):
            if descriptor_idx < len(raw_image_bytes):
                descriptor["_raw_image_bytes"] = raw_image_bytes[descriptor_idx]
            if descriptor.get("kind") == "raw_bytes" or "_raw_image_bytes" in descriptor:
                descriptor.setdefault("materializer", _RAW_JPGS_MATERIALIZER)
            grid = self._grid_from_descriptor(descriptor)
            t, h, w = [int(x) for x in grid]
            tokens = t * (h // self.spatial_merge_size) * (w // self.spatial_merge_size)
            this_cost = 2 + tokens
            if image_block_cost + this_cost > self.seq_length:
                if not image_token_counts:
                    raise ValueError(
                        "single image token span exceeds seq_length: "
                        f"image_tokens={this_cost}, seq_length={self.seq_length}"
                    )
                break
            descriptor["grid_thw"] = [t, h, w]
            descriptor.setdefault("pixel_dim", int(self._pixel_dim))
            descriptor.setdefault("temporal_patch_size", int(self.temporal_patch_size))
            descriptor.setdefault("spatial_merge_size", int(self.spatial_merge_size))
            image_grids.append([t, h, w])
            image_patch_counts.append(t * h * w)
            image_token_counts.append(tokens)
            image_block_cost += this_cost

        (input_ids, assistant_mask, included_images) = self._chat_template_to_tokens(
            turns, image_token_counts
        )
        if int(input_ids.numel()) <= 0:
            raise ValueError("empty qwen35 Energon sample")
        if included_images < len(image_token_counts):
            image_grids = image_grids[:included_images]
            image_patch_counts = image_patch_counts[:included_images]
            image_token_counts = image_token_counts[:included_images]

        if image_grids:
            image_grid_thw = torch.tensor(image_grids, dtype=torch.long)
        else:
            image_grid_thw = torch.zeros(0, 3, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "_assistant_loss_mask": assistant_mask,
            "real_len": int(input_ids.numel()),
            "content_len": int(input_ids.numel()),
            "pixel_values": torch.zeros(0, self._pixel_dim, dtype=torch.float32),
            "image_grid_thw": image_grid_thw,
            "num_images": len(image_grids),
            "num_patches": int(sum(image_patch_counts)),
            "_mdp_image_descriptors": descriptors[: len(image_grids)],
        }

    def select_samples_to_pack(self, samples):
        groups = []
        current = []
        used = 0
        for sample in samples:
            padded = self._pad_doc(sample)
            if padded is None:
                continue
            length = int(padded["real_len"])
            if length <= 0 or length > self.seq_length:
                continue
            if current and used + length > self.seq_length:
                groups.append(current)
                current = []
                used = 0
            current.append(padded)
            used += length
            if used >= self.seq_length:
                groups.append(current)
                current = []
                used = 0
        if current:
            groups.append(current)
        self._attach_window_assignments(groups)
        return groups

    def _attach_window_assignments(self, groups):
        """Attach one deterministic CP owner to each selected image."""
        if not self.mdp_loader_prepartition or not self.mdp_loader_prepartition_materialize:
            return
        image_refs = []
        costs_by_item = []
        for group in groups:
            refs = []
            costs = []
            for doc_idx, doc in enumerate(group):
                for image_idx, cost in enumerate(self._image_balance_costs(doc)):
                    refs.append((doc_idx, image_idx))
                    costs.append(cost)
            image_refs.append(refs)
            costs_by_item.append(costs)

        assignment = assign_images_lpt(
            costs_by_item,
            self.mdp_loader_prepartition_world,
            across_items=False,
        )
        for group in groups:
            for doc in group:
                doc["_mdp_image_owner_ranks"] = [-1] * int(doc["num_images"])
        for rank, items in assignment.items():
            for item_idx, image_idx in items:
                doc_idx, local_image_idx = image_refs[int(item_idx)][int(image_idx)]
                groups[int(item_idx)][doc_idx]["_mdp_image_owner_ranks"][
                    local_image_idx
                ] = int(rank)

    def _assignment_from_doc_owners(self, docs: Sequence[dict]):
        assignment = {
            rank: [] for rank in range(self.mdp_loader_prepartition_world)
        }
        for doc_idx, doc in enumerate(docs):
            owners = doc.get("_mdp_image_owner_ranks")
            if owners is None:
                single_assignment = assign_images_lpt(
                    [self._image_balance_costs(doc)],
                    self.mdp_loader_prepartition_world,
                    across_items=False,
                )
                owners = [-1] * int(doc["num_images"])
                for rank, items in single_assignment.items():
                    for _item_idx, image_idx in items:
                        owners[int(image_idx)] = int(rank)
            for image_idx, owner in enumerate(owners):
                if int(owner) >= 0:
                    assignment[int(owner)].append((int(doc_idx), int(image_idx)))
        return assignment

    def _image_balance_costs(self, doc: dict):
        grid = doc.get("image_grid_thw")
        if not torch.is_tensor(grid) or grid.numel() == 0:
            return []
        return image_costs_from_grid(
            grid.detach().cpu().tolist(),
            hidden_size=self.mdp_lpt_hidden_size,
        )

    def _prepartition_assignment_tensor(self, assignment, doc_offsets):
        rows = []
        for rank in sorted(assignment):
            for doc_idx, image_idx in assignment[rank]:
                rows.append(
                    [
                        int(rank),
                        int(doc_idx),
                        int(doc_offsets[int(doc_idx)] + int(image_idx)),
                    ]
                )
        if not rows:
            return torch.zeros(0, 3, dtype=torch.int32)
        return torch.tensor(rows, dtype=torch.int32)

    def _attach_prepartition(self, out: dict, docs: Sequence[dict]) -> dict:
        """Materialize only this CP rank's LPT-owned image descriptors."""
        if not self.mdp_loader_prepartition:
            return out
        world = self.mdp_loader_prepartition_world
        rank = self.mdp_loader_prepartition_rank
        if rank < 0 or rank >= world:
            raise RuntimeError(
                "loader prepartition rank is outside its CP world: "
                f"rank={rank} world={world}"
            )

        assignment = self._assignment_from_doc_owners(docs)
        doc_offsets = []
        offset = 0
        for doc in docs:
            doc_offsets.append(offset)
            offset += int(doc["num_images"])

        local_pixels = []
        local_grids = []
        if self.mdp_loader_prepartition_encoder_stage and self.mdp_loader_prepartition_materialize:
            for doc_idx, image_idx in assignment.get(rank, []):
                doc = docs[int(doc_idx)]
                grid = [
                    int(value)
                    for value in doc["image_grid_thw"][int(image_idx)]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                descriptor = doc["_mdp_image_descriptors"][int(image_idx)]
                patches = materialize_descriptor(
                    descriptor,
                    grid,
                    pixel_dim=int(self._pixel_dim),
                    patch_size=int(self.patch_size),
                )
                local_pixels.append(patches.to(torch.bfloat16))
                local_grids.append(torch.tensor(grid, dtype=torch.long))

        if local_pixels:
            out["pixel_values"] = torch.cat(local_pixels, dim=0)
            local_grid = torch.stack(local_grids, dim=0)
        else:
            out["pixel_values"] = torch.zeros(
                0, self._pixel_dim, dtype=torch.bfloat16
            )
            local_grid = torch.zeros(0, 3, dtype=torch.long)

        global_rows = []
        for doc in docs:
            for row in doc["image_grid_thw"].detach().cpu().tolist():
                global_rows.append(
                    vision_rows_from_grid(row, self.spatial_merge_size)
                )
        out["_mdp_prepartitioned_image_grid_thw"] = local_grid
        out["_mdp_prepartitioned_assignment"] = (
            self._prepartition_assignment_tensor(assignment, doc_offsets)
        )
        out["_mdp_prepartitioned_row_counts"] = torch.tensor(
            global_rows, dtype=torch.int32
        )
        return out

    def _materialize_all_images(self, descriptors, image_grid_thw) -> torch.Tensor:
        local_pixels = []
        rows = image_grid_thw.detach().cpu().tolist()
        if len(descriptors) != len(rows):
            raise RuntimeError(
                "qwen35 Energon full materialization descriptor/grid mismatch: "
                f"descriptors={len(descriptors)} grids={len(rows)}"
            )
        for descriptor, grid in zip(descriptors, rows):
            patches = materialize_descriptor(
                descriptor,
                [int(x) for x in grid],
                pixel_dim=int(self._pixel_dim),
                patch_size=int(self.patch_size),
            )
            local_pixels.append(patches.to(torch.float32))
        if not local_pixels:
            return torch.zeros(0, self._pixel_dim, dtype=torch.float32)
        return torch.cat(local_pixels, dim=0)

    @stateless
    def pack_selected_samples(self, samples):
        if not samples:
            raise RuntimeError("cannot pack an empty qwen35 Energon group")
        padded_docs = []
        used = 0
        for sample in samples:
            padded = self._pad_doc(sample, max_length=self.seq_length - used)
            if padded is None:
                raise RuntimeError(
                    f"packed sample does not fit remaining sequence budget {self.seq_length - used}"
                )
            padded_docs.append(padded)
            used += int(padded["real_len"])
        docs = padded_docs
        real_lens = [int(doc["real_len"]) for doc in docs]
        content_lens = [int(doc.get("content_len", doc["real_len"])) for doc in docs]
        real_total = sum(real_lens)
        pad_len = self.seq_length - real_total
        if pad_len < 0:
            raise RuntimeError(
                f"packed unpadded length {real_total} exceeds seq_length {self.seq_length}"
            )

        input_ids_real = torch.cat([doc["input_ids"] for doc in docs], dim=0)
        assistant_mask_real = torch.cat([doc["_assistant_loss_mask"] for doc in docs], dim=0)
        input_ids = (
            torch.cat([input_ids_real, torch.zeros(pad_len, dtype=torch.long)])
            if pad_len > 0
            else input_ids_real
        )
        assistant_mask = (
            torch.cat([assistant_mask_real, torch.zeros(pad_len, dtype=torch.float32)])
            if pad_len > 0
            else assistant_mask_real
        )
        labels, loss_mask = self._shifted_labels_and_loss_mask(
            input_ids, assistant_mask, real_lens, content_lens
        )

        grid_parts = [
            doc["image_grid_thw"] for doc in docs if int(doc["image_grid_thw"].shape[0]) > 0
        ]
        image_grid_thw = (
            torch.cat(grid_parts, dim=0) if grid_parts else torch.zeros(0, 3, dtype=torch.long)
        )

        position_parts = [
            self._position_ids_for_doc(doc["input_ids"], doc["image_grid_thw"]) for doc in docs
        ]
        position_ids_real = torch.cat(position_parts, dim=1)
        if pad_len > 0:
            if position_ids_real.shape[1] == 0:
                pad_positions = torch.arange(pad_len, dtype=torch.long).repeat(3, 1)
            else:
                increments = torch.arange(1, pad_len + 1, dtype=position_ids_real.dtype).view(1, -1)
                pad_positions = position_ids_real[:, -1:] + increments
            position_ids = torch.cat([position_ids_real, pad_positions], dim=1)
        else:
            position_ids = position_ids_real

        descriptors = []
        for doc in docs:
            descriptors.extend(doc.get("_mdp_image_descriptors", []))

        pixel_values = torch.zeros(0, self._pixel_dim, dtype=torch.float32)
        if not self.mdp_loader_prepartition:
            pixel_values = self._materialize_all_images(descriptors, image_grid_thw)

        out = {
            "input_ids": input_ids,
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_images": int(sum(int(doc["num_images"]) for doc in docs)),
            "_mdp_image_descriptors_json": encode_image_descriptors(descriptors),
        }
        if self.mdp_loader_prepartition and not self.mdp_loader_prepartition_materialize:
            out["_mdp_image_descriptors"] = descriptors
        if self.mdp_loader_prepartition_materialize:
            out = self._attach_prepartition(out, docs)

        cu = [0]
        for doc_len in real_lens:
            cu.append(cu[-1] + doc_len)
        cu_padded = list(cu)
        cu_padded[-1] = self.seq_length
        image_cu = [0]
        pixel_cu = [0]
        for doc in docs:
            image_cu.append(image_cu[-1] + int(doc["num_images"]))
            pixel_cu.append(pixel_cu[-1] + int(doc["num_patches"]))
        padded_seg_lens = [cu_padded[i + 1] - cu_padded[i] for i in range(len(cu_padded) - 1)]
        out["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32)
        out["cu_seqlens_padded"] = torch.tensor(cu_padded, dtype=torch.int32)
        out["max_seqlen"] = torch.tensor(max(padded_seg_lens), dtype=torch.int32)
        out["image_cu_seqlens"] = torch.tensor(image_cu, dtype=torch.int32)
        out["pixel_cu_seqlens"] = torch.tensor(pixel_cu, dtype=torch.int32)
        return out

    @stateless
    def batch(self, samples):
        if len(samples) != 1:
            raise RuntimeError("qwen35 Energon path expects batch_size=1 after packing")
        return samples[0]

    @stateless
    def encode_batch(self, batch):
        return batch
