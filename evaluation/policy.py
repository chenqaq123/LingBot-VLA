"""Official inference methods, with streaming weight loading and two-GPU placement.

Run directly from the workspace upstream checkout. The only overrides concern allocation of checkpoint tensors.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
import torch

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
from deploy.websocket_policy_server import WebsocketPolicyServer


class OfficialPolicy(LingbotVLAv2Server):
    def __init__(self, checkpoint, frontend_device=None, use_bf16=False):
        # Same wrapper state as upstream __init__. Placement replaces its final
        # whole-model .cuda(), which cannot fit FP32 weights on a 24 GB card.
        self.adaptive_ensemble_alpha = 0.1
        self.action_ensemble_horizon = 8
        self.use_length = 50
        self.chunk_ret = True
        self.robot_norm_path = None
        self.task_description = None
        self.use_compile = False
        self.use_bf16 = use_bf16
        self.use_fp32 = not use_bf16
        self.primary_device = torch.device('cuda:0')
        self.frontend_device = torch.device(frontend_device or 'cuda:0')
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
        apply_lingbot_qwen3_vl_patch()
        with init_empty_weights():
            self.vla = super().load_vla(str(checkpoint))
        self.vla.eval()
        self.global_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None
        self.action_key = 'action'
        self._place_buffers_and_bridge()
        self.reset('robotwin')

    def _device_for(self, name):
        prefixes = (
            'model.qwenvl_with_expert.qwenvl.model.visual.',
            'model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.',
        )
        return self.frontend_device if name.startswith(prefixes) else self.primary_device

    def load_model_weights(self, path_to_pi_model, strict=True):
        expected = set(self.vla.state_dict())
        loaded = set()
        dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        shards = sorted(Path(path_to_pi_model).glob('*.safetensors'))
        if not shards:
            raise ValueError('A full safetensors checkpoint is required; incremental adapters are unsupported.')
        for shard_path in shards:
            print('Loading official checkpoint:', shard_path.name, flush=True)
            with safe_open(shard_path, framework='pt', device='cpu') as shard:
                for name in shard.keys():
                    if name not in expected or name in loaded:
                        raise ValueError(f'Unexpected or duplicate checkpoint tensor: {name}')
                    set_module_tensor_to_device(self.vla, name, self._device_for(name),
                                                value=shard.get_tensor(name), dtype=dtype)
                    loaded.add(name)
        if loaded != expected:
            raise ValueError(f'Incomplete checkpoint, missing: {sorted(expected-loaded)}')
        self.loaded_tensor_count = len(loaded)

    def _place_buffers_and_bridge(self):
        dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        for name, value in list(self.vla.named_buffers()):
            set_module_tensor_to_device(self.vla, name, self._device_for(name), value=value,
                                        dtype=dtype if value.is_floating_point() else value.dtype)
        if self.frontend_device == self.primary_device:
            return
        bridge = self.vla.model.qwenvl_with_expert
        embed_image, embed_language = bridge.embed_image, bridge.embed_language_tokens
        def image_on_frontend(image, grid):
            with torch.cuda.device(self.frontend_device):
                features, deepstack = embed_image(image.to(self.frontend_device), grid.to(self.frontend_device))
            return features.to(self.primary_device), [x.to(self.primary_device) for x in deepstack]
        def language_on_frontend(tokens):
            with torch.cuda.device(self.frontend_device):
                return embed_language(tokens.to(self.frontend_device)).to(self.primary_device)
        bridge.embed_image = image_on_frontend
        bridge.embed_language_tokens = language_on_frontend

    def reset(self, robo_name, path_to_pi_model=None):
        if path_to_pi_model is not None:
            raise ValueError('Restart the server to load a different checkpoint.')
        return super().reset(robo_name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--ready-file', required=True)
    p.add_argument('--frontend-device')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    args = p.parse_args()
    torch.set_num_threads(4)
    policy = OfficialPolicy(args.checkpoint, args.frontend_device, args.precision == 'bf16')
    norm = Path(policy.robot_norm_path or policy.robot_config['norm_stats']).resolve()
    metadata = {
        'implementation': 'official', 'checkpoint': args.checkpoint,
        'precision_mode': args.precision, 'native_queries': bool(policy.config.align_params),
        'loaded_tensors': policy.loaded_tensor_count,
        'parameter_dtypes': dict(Counter(str(x.dtype) for x in policy.vla.parameters())),
        'normalization': str(norm), 'norm_sha256': hashlib.sha256(norm.read_bytes()).hexdigest(),
        'image_size': getattr(policy.data_config, 'img_size', 256), 'color': 'rgb',
        'vision_attention': policy.config.vit_attn_implementation,
        'joint_attention': policy.config.attention_implementation,
        'chunk_size': policy.config.chunk_size, 'execute_steps': policy.use_length,
        'denoising_steps': policy.config.num_steps, 'compile': False,
        'upstream_root': str(Path.cwd()),
        'device_parameter_bytes': {str(d): sum(p.numel()*p.element_size() for p in policy.vla.parameters() if p.device == d)
                                   for d in {p.device for p in policy.vla.parameters()}},
        'precision_note': 'Official FP32 mode preserves upstream FlashAttention BF16 casts and kernel defaults; not strict all-operation FP32.',
    }
    Path(args.ready_file).write_text(json.dumps(metadata, indent=2))
    print('READY', json.dumps(metadata), flush=True)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
