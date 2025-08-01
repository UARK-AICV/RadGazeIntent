from typing import Optional

import einops
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone
from torch import Tensor

import common.position_encoding as pe

from .config import add_maskformer2_config
from .pixel_decoder.fpn import TransformerEncoderPixelDecoder
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder,
)
from .transformer_decoder.transformer import TransformerEncoder, TransformerEncoderLayer


def attention_pool(tensor, pool, norm=None):
    if pool is None:
        return tensor
    L, B, D = tensor.shape
    tensor = einops.rearrange(tensor, "l b d -> b d l")
    tensor = pool(tensor)
    tensor = einops.rearrange(tensor, "b d l -> l b d")
    if norm is not None:
        tensor = norm(tensor)
    return tensor


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long())  # * math.sqrt(self.emb_size)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class ImageFeatureEncoder(nn.Module):
    def __init__(
        self,
        cfg_path,
        dropout,
        pixel_decoder="MSD",
        load_segm_decoder=False,
        pred_saliency=False,
    ):
        super(ImageFeatureEncoder, self).__init__()

        # Load Detectrion2 backbone
        cfg = get_cfg()
        add_maskformer2_config(cfg)
        cfg.merge_from_file(cfg_path)
        self.backbone = build_backbone(cfg)
        # if os.path.exists(cfg.MODEL.WEIGHTS):
        bb_weights = torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))
        bb_weights_new = bb_weights.copy()
        for k, v in bb_weights.items():
            if k[:7] == "stages.":
                bb_weights_new[k[7:]] = v
                bb_weights_new.pop(k)
        self.backbone.load_state_dict(bb_weights_new)
        self.backbone.eval()
        print("Loaded backbone weights from {}".format(cfg.MODEL.WEIGHTS))

        if pred_saliency:
            assert not load_segm_decoder, (
                "cannot load segmentation decoder and predict saliency at the same time"
            )
            self.saliency_head = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, padding=0),
            )
        else:
            self.saliency_head = None

        # Load deformable pixel decoder
        if cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
            input_shape = {
                "res2": ShapeSpec(channels=128, stride=4),
                "res3": ShapeSpec(channels=256, stride=8),
                "res4": ShapeSpec(channels=512, stride=16),
                "res5": ShapeSpec(channels=1024, stride=32),
            }
        else:
            input_shape = {
                "res2": ShapeSpec(channels=256, stride=4),
                "res3": ShapeSpec(channels=512, stride=8),
                "res4": ShapeSpec(channels=1024, stride=16),
                "res5": ShapeSpec(channels=2048, stride=32),
            }
        args = {
            "input_shape": input_shape,
            "conv_dim": 256,
            "mask_dim": 256,
            "norm": "GN",
            "transformer_dropout": dropout,
            "transformer_nheads": 8,
            "transformer_dim_feedforward": 1024,
            "transformer_enc_layers": 6,
            "transformer_in_features": ["res3", "res4", "res5"],
            "common_stride": 4,
        }
        if pixel_decoder == "MSD":
            msd = MSDeformAttnPixelDecoder(**args)
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + "_MSDeformAttnPixelDecoder.pkl"
            # if os.path.exists(ckpt_path):
            msd_weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
            msd_weights_new = msd_weights.copy()
            for k, v in msd_weights.items():
                if k[:7] == "adapter":
                    msd_weights_new["lateral_convs." + k] = v
                    msd_weights_new.pop(k)
                elif k[:5] == "layer":
                    msd_weights_new["output_convs." + k] = v
                    msd_weights_new.pop(k)
            msd.load_state_dict(msd_weights_new)
            print("Loaded MSD pixel decoder weights from {}".format(ckpt_path))
            self.pixel_decoder = msd
            self.pixel_decoder.eval()
        elif pixel_decoder == "FPN":
            args.pop("transformer_in_features")
            args.pop("common_stride")
            args["transformer_dim_feedforward"] = 2048
            args["transformer_pre_norm"] = False
            fpn = TransformerEncoderPixelDecoder(**args)
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + "_FPN.pkl"
            # if os.path.exists(ckpt_path):
            fpn_weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
            fpn.load_state_dict(fpn_weights)
            self.pixel_decoder = fpn
            print("Loaded FPN pixel decoder weights from {}".format(ckpt_path))
            self.pixel_decoder.eval()
        else:
            raise NotImplementedError

        # Load segmentation decoder
        self.load_segm_decoder = load_segm_decoder
        if self.load_segm_decoder:
            args = {
                "in_channels": 256,
                "mask_classification": True,
                "num_classes": 133,
                "hidden_dim": 256,
                "num_queries": 100,
                "nheads": 8,
                "dim_feedforward": 2048,
                "dec_layers": 9,
                "pre_norm": False,
                "mask_dim": 256,
                "enforce_input_project": False,
            }
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + "_transformer_decoder.pkl"
            mtd = MultiScaleMaskedTransformerDecoder(**args)
            mtd_weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
            mtd.load_state_dict(mtd_weights)
            self.segm_decoder = mtd
            print("Loaded segmentation decoder weights from {}".format(ckpt_path))
            self.segm_decoder.eval()

    def forward(self, x):
        features = self.backbone(x)
        high_res_featmaps, _, ms_feats = self.pixel_decoder.forward_features(features)
        if self.load_segm_decoder:
            segm_predictions = self.segm_decoder.forward(ms_feats, high_res_featmaps)
            queries = segm_predictions["out_queries"]

            segm_results = self.segmentation_inference(segm_predictions)
            # segm_results = None
            return high_res_featmaps, queries, segm_results
        else:
            if self.saliency_head is not None:
                saliency_map = self.saliency_head(high_res_featmaps)
                return {"pred_saliency": saliency_map}
            else:
                return high_res_featmaps, ms_feats[0], ms_feats[1]

    def segmentation_inference(self, segm_preds):
        """Compute panoptic segmentation from the outputs of the segmentation decoder."""
        mask_cls_results = segm_preds.pop("pred_logits")
        mask_pred_results = segm_preds.pop("pred_masks")

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(
            mask_cls_results, mask_pred_results
        ):
            panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result)
            processed_results.append(panoptic_r)

        return processed_results

    def panoptic_inference(
        self, mask_cls, mask_pred, object_mask_threshold=0.8, overlap_threshold=0.8
    ):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        # Remove non-object masks and masks with low confidence
        keep = labels.ne(mask_cls.size(-1) - 1) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        keep_ids = torch.where(keep)[0]

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return [], [], keep
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in range(80)
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        keep[keep_ids[k]] = False
                        continue

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    my, mx = torch.where(mask)
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                            "mask_area": mask_area,
                            "mask_centroid": (mx.float().mean(), my.float().mean()),
                        }
                    )
                else:
                    keep[keep_ids[k]] = False

            return panoptic_seg, segments_info, keep


# Dense prediction transformer
class HumanAttnTransformer(nn.Module):
    def __init__(
        self,
        pa,
        num_decoder_layers: int,
        hidden_dim: int,
        nhead: int,
        num_output_layers: int,
        catIds,
        train_encoder: bool = False,
        train_foveator: bool = True,
        train_pixel_decoder: bool = False,
        pre_norm: bool = False,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        parallel_arch: bool = False,
        dorsal_source: list = ["P2"],
        num_encoder_layers: int = 3,
        output_centermap: bool = False,
        output_saliency: bool = False,
        output_target_map: bool = False,
        combine_pos_emb: bool = True,
        combine_all_emb: bool = False,
        project_queries: bool = True,
        is_pretraining: bool = False,
        output_feature_map_name: str = "P4",
    ):
        super(HumanAttnTransformer, self).__init__()
        self.pa = pa
        self.num_decoder_layers = num_decoder_layers
        self.is_pretraining = is_pretraining
        self.combine_pos_emb = combine_pos_emb
        self.combine_all_emb = combine_all_emb
        self.output_feature_map_name = output_feature_map_name
        self.parallel_arch = parallel_arch
        self.dorsal_source = dorsal_source
        assert len(dorsal_source) > 0, "need to specify dorsal source: P1, P2!"
        self.output_centermap = output_centermap
        self.output_saliency = output_saliency
        self.output_target_map = output_target_map

        # Encoder: Deformable Attention Transformer
        self.train_encoder = train_encoder
        self.encoder = ImageFeatureEncoder(
            pa.backbone_config, dropout, pa.pixel_decoder
        )
        self.symbol_offset = len(self.pa.special_symbols)
        if not train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.train_pixel_decoder = train_pixel_decoder
        if train_pixel_decoder:
            self.encoder.pixel_decoder.train()
            for param in self.encoder.pixel_decoder.parameters():
                param.requires_grad = True
        featmap_channels = 256
        if hidden_dim != featmap_channels:
            self.input_proj = nn.Conv2d(featmap_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()

        # Transfer learning setting (only support COCO-Search18 for now, where
        # we assume 18 search targets).
        self.project_queries = project_queries

        self.num_encoder_layers = num_encoder_layers
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
        self.working_memory_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        if not train_foveator:
            self.working_memory_encoder.eval()
            for param in self.working_memory_encoder.parameters():
                param.requires_grad = False

        # Pooling
        dim_conv = hidden_dim
        self.norm_q = nn.LayerNorm(dim_conv)
        self.pool_q = nn.Conv1d(
            dim_conv,
            dim_conv,
            3,
            stride=1,
            padding=0,
            groups=dim_conv,
            bias=False,
        )
        # Decoder
        _hw = 384
        self.ntask = len(catIds)
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_dorsal = nn.ModuleList()
        self.transformer_cross_attention_layers_ventral = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers_dorsal.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            if not self.parallel_arch:
                self.transformer_cross_attention_layers_ventral.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    )
                )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.logit_layer = MLP(_hw, _hw, self.ntask, num_output_layers)
        # Positional embedding
        self.pixel_loc_emb = pe.PositionalEncoding2D(
            pa, hidden_dim, height=pa.im_h // 4, width=pa.im_w // 4, dropout=dropout
        )
        if self.output_feature_map_name == "P4":
            self.pos_scale = 1
        elif self.output_feature_map_name == "P2":
            self.pos_scale = 4
        else:
            raise NotImplementedError

        self.fix_ind_emb = nn.Embedding(pa.max_traj_length, hidden_dim)

        # Embedding for distinguishing dorsal or ventral embeddings
        self.dorsal_ind_emb = nn.Embedding(2, hidden_dim)  # P1 and P2
        self.ventral_ind_emb = nn.Embedding(1, hidden_dim)

    def forward(
        self,
        img: torch.Tensor,
        tgt_seq: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        tgt_seq_high: torch.Tensor,
        return_attn_weights=False,
    ):
        # Prepare dorsal embeddings
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        output_featmaps = high_res_featmaps

        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        # c x 10 x 16
        img_embs = self.input_proj(img_embs_s1)
        bs, c, h, w = img_embs.shape
        pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:], scale=8)
        img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
        scale_embs.append(
            self.dorsal_ind_emb.weight[0]
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(img_embs.size(0), bs, c)
        )
        dorsal_embs.append(img_embs)
        dorsal_pos.append(pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)

        bs = high_res_featmaps.size(0)
        # Prepare ventral embeddings
        if tgt_seq_high is None:
            tgt_seq = tgt_seq.transpose(0, 1)
            ventral_embs = torch.gather(
                torch.cat(
                    [
                        torch.zeros(1, *img_embs.shape[1:], device=img_embs.device),
                        img_embs,
                    ],
                    dim=0,
                ),
                0,
                tgt_seq.unsqueeze(-1).expand(*tgt_seq.shape, img_embs.size(-1)),
            )
            ventral_pos = self.pixel_loc_emb(tgt_seq)  # Pos for fixation location
        else:
            tgt_seq_high = tgt_seq_high.transpose(0, 1)
            highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
            ventral_embs = torch.gather(
                torch.cat(
                    [
                        torch.zeros(1, *highres_embs.shape[1:], device=img_embs.device),
                        highres_embs,
                    ],
                    dim=0,
                ),
                0,
                tgt_seq_high.unsqueeze(-1).expand(
                    *tgt_seq_high.shape, highres_embs.size(-1)
                ),
            )
            # Pos for fixation location
            ventral_pos = self.pixel_loc_emb(tgt_seq_high)

        # Add pos into embeddings for attention prediction
        if self.combine_pos_emb:
            # Dorsal embeddings
            dorsal_embs += dorsal_pos
            dorsal_pos.fill_(0)
            # Ventral embeddings
            ventral_embs += ventral_pos
            ventral_pos.fill_(0)
            output_featmaps += self.pixel_loc_emb.forward_featmaps(
                output_featmaps.shape[-2:], scale=self.pos_scale
            )

        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(
            *ventral_pos.shape
        )

        ventral_pos += (
            self.fix_ind_emb.weight[: ventral_embs.size(0)]
            .unsqueeze(1)
            .repeat(1, bs, 1)
        )
        ventral_pos[tgt_padding_mask.transpose(0, 1)] = 0
        # shape working_memory = (55 = 49 + 6, bs, 384)
        working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
        padding_mask = torch.cat(
            [
                torch.zeros(bs, dorsal_embs.size(0), device=dorsal_embs.device).bool(),
                tgt_padding_mask,
            ],
            dim=1,
        )
        working_memory_pos = torch.cat([dorsal_pos, ventral_pos], dim=0)

        working_memory = self.working_memory_encoder(
            working_memory,
            src_key_padding_mask=padding_mask,
            pos=working_memory_pos,
        )
        working_memory_cloned = working_memory.clone()
        working_memory_pos_cloned = working_memory_pos.clone()
        out = {}
        # shape out  = L/2, B, D
        working_memory_pooled = attention_pool(working_memory, self.pool_q, self.norm_q)
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            query_embed, attn_weights = self.transformer_cross_attention_layers_dorsal[
                i
            ](
                working_memory_cloned,
                working_memory_pooled,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None,
            )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        outputs_logit = self.logit_layer(query_embed[dorsal_embs.size(0) :])
        outputs_logit = einops.rearrange(outputs_logit, "l b d -> b l d")
        if self.training:
            out["outputs_logit"] = outputs_logit
        else:
            out["outputs_logit"] = outputs_logit
        return out

    def encode(self, img: torch.Tensor):
        # Prepare dorsal embeddings
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        output_featmaps = high_res_featmaps
        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        # C x 10 x 16
        img_embs = self.input_proj(img_embs_s1)
        bs, c, h, w = img_embs.shape
        pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:], scale=8)
        img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
        scale_embs.append(
            self.dorsal_ind_emb.weight[0]
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(img_embs.size(0), bs, c)
        )
        dorsal_embs.append(img_embs)
        dorsal_pos.append(pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)
        dorsal_pos = (dorsal_pos, scale_embs)

        return dorsal_embs, dorsal_pos, None, (high_res_featmaps, output_featmaps)
