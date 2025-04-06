# sam2_model.py
import torch
from SAM2.sam2.training.model.sam2 import SAM2Train
from SAM2.sam2.sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from SAM2.sam2.sam2.modeling.backbones.hieradet import Hiera
from SAM2.sam2.sam2.modeling.position_encoding import PositionEmbeddingSine
from SAM2.sam2.sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from SAM2.sam2.sam2.modeling.sam.transformer import RoPEAttention
from SAM2.sam2.sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock

def build_base_model():
    trunk = Hiera(embed_dim=112, num_heads=2)

    position_encoding_fpn = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, temperature=10000
    )

    neck = FpnNeck(
        position_encoding=position_encoding_fpn,
        d_model=256,
        backbone_channel_list=[896, 448, 224, 112],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model='nearest'
    )

    image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)

    self_attn = RoPEAttention(
        rope_theta=10000.0, feat_sizes=[64, 64], embedding_dim=256,
        num_heads=1, downsample_rate=1, dropout=0.1
    )

    cross_attn = RoPEAttention(
        rope_theta=10000.0, rope_k_repeat=True, feat_sizes=[64, 64],
        embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1, kv_in_dim=64
    )

    memory_attention_layer = MemoryAttentionLayer(
        activation='relu', dim_feedforward=2048, dropout=0.1,
        pos_enc_at_attn=False, d_model=256,
        pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        self_attention=self_attn, cross_attention=cross_attn
    )

    memory_attention = MemoryAttention(
        d_model=256, pos_enc_at_input=True,
        layer=memory_attention_layer, num_layers=4
    )

    pos_enc_mem = PositionEmbeddingSine(num_pos_feats=64, normalize=True, temperature=10000)

    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    fuser = Fuser(CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True), num_layers=2)

    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=pos_enc_mem,
        mask_downsampler=mask_downsampler,
        fuser=fuser
    )

    model = SAM2Train(
    image_encoder,
    memory_attention=memory_attention,
    memory_encoder=memory_encoder,
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    use_high_res_features_in_sam=True,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=True,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    no_obj_embed_spatial = True,
    proj_tpos_enc_in_obj_ptrs = True,
    compile_image_encoder=False)

    return model

def build_small_model():
    trunk = Hiera(
        embed_dim=96,
        num_heads=1,
        stages=[1, 2, 11, 2],
        global_att_blocks=[7, 10, 13],
        window_pos_embed_bkg_spatial_size=[7, 7]
    )

    position_encoding_fpn = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, temperature=10000, scale=None
    )

    neck = FpnNeck(
        position_encoding=position_encoding_fpn,
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model='nearest'
    )

    image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)

    self_attn = RoPEAttention(
        rope_theta=10000.0, feat_sizes=[64, 64], embedding_dim=256,
        num_heads=1, downsample_rate=1, dropout=0.1
    )

    cross_attn = RoPEAttention(
        rope_theta=10000.0, rope_k_repeat=True, feat_sizes=[64, 64],
        embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1, kv_in_dim=64
    )

    memory_attention_layer = MemoryAttentionLayer(
        activation='relu', dim_feedforward=2048, dropout=0.1,
        pos_enc_at_attn=False, d_model=256,
        pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        self_attention=self_attn, cross_attention=cross_attn
    )

    memory_attention = MemoryAttention(
        d_model=256, pos_enc_at_input=True,
        layer=memory_attention_layer, num_layers=4
    )

    pos_enc_mem = PositionEmbeddingSine(num_pos_feats=64, normalize=True, temperature=10000, scale=None)


    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    fuser = Fuser(CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True), num_layers=2)

    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=pos_enc_mem,
        mask_downsampler=mask_downsampler,
        fuser=fuser
    )
    
    model = SAM2Train(
    image_encoder,
    memory_attention=memory_attention,
    memory_encoder=memory_encoder,
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    use_high_res_features_in_sam=True,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=True,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    no_obj_embed_spatial = True,
    proj_tpos_enc_in_obj_ptrs = True,
    compile_image_encoder=False)

    return model

def get_model(cfg):
    if "base" in cfg.model.checkpoint:
        model = build_base_model()
    elif "small" in cfg.model.checkpoint:
        model = build_small_model()
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    return model

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        device = next(model.parameters()).device
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            raise RuntimeError()
        if unexpected_keys:
            raise RuntimeError()



