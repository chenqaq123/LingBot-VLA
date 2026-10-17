import copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch,yaml
upstream=Path(__file__).resolve().parents[1]/"lingbot-vla-v2"
sys.path.insert(0,str(upstream))
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from lingbotvla.models import build_processor
from lingbotvla.data.vla_data.utils import FeatureTransform
cfg=yaml.safe_load((upstream/'configs/vla/robotwin/robotwin.yaml').read_text())
for key in ['joints','norm_type']:cfg['data'][key]=[repr(x) for x in cfg['data'][key]]
config=LingbotVLAV2Config(**{**cfg['model'],**cfg['train']})
processor=build_processor('/srv/common_model/huggingface/Qwen3-VL-4B-Instruct')
normpath=upstream/'assets/norm_stats/robotwin.json'
transform=FeatureTransform(str(upstream/'configs/robot_configs/robotwin.yaml'),SimpleNamespace(**cfg['data']),config,processor,norm_stats_path=str(normpath))
policy=LingbotVLAv2Server.__new__(LingbotVLAv2Server)
policy.vla=SimpleNamespace(feature_transform=transform)
policy.data_config=SimpleNamespace(**cfg['data']);policy.use_bf16=False
state=np.linspace(-1,1,14,dtype=np.float32);state[6]=.25;state[13]=.75
obs={'task':'Pick up the red bottle.','observation.state':state.copy()}
for i,name in enumerate(['cam_high','cam_left_wrist','cam_right_wrist']):
 image=np.zeros((240,320,3),dtype=np.uint8);image[:,:,i]=255
 obs['observation.images.'+name]=image
resized=copy.deepcopy(obs);policy.resize_image(resized)
for i,name in enumerate(['cam_high','cam_left_wrist','cam_right_wrist']):
 image=resized['observation.images.'+name]
 assert tuple(image.shape)==(3,256,256)
 assert torch.allclose(image[i],torch.full_like(image[i],255),atol=.001)
 assert all(torch.count_nonzero(image[j])==0 for j in range(3) if j!=i)
batch=policy._prepare_model_input(obs)
assert np.array_equal(obs['observation.state'],state)
ns=json.loads(normpath.read_text())['norm_stats']
expected=np.zeros(55,dtype=np.float32)
for key,raw,start in [('arm.position',np.r_[state[:6],state[7:13]],0),('effector.position',state[[6,13]],28)]:
 stats=ns['observation.state.'+key];lo=np.asarray(stats['q01'],np.float32);hi=np.asarray(stats['q99'],np.float32)
 expected[start:start+len(raw)]=(raw-lo)/(hi-lo+1e-6)*2-1
assert np.allclose(batch['state'].numpy(),expected,atol=1e-6)
assert batch['lang_tokens'].numel()==72
assert batch['img_masks'].all()
report={'passed':True,'resize':[256,256],'rgb_channel_check':True,'state_55d_layout_and_normalization':True,'language_token_length':72,'normalization_sha256':hashlib.sha256(normpath.read_bytes()).hexdigest(),'tensor_shapes':{k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)}}
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
Path(args.output).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
