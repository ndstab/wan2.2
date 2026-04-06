import torch
import decord
from wan.wsis import WeatherStyleInjectionScore

device = torch.device("cuda")
scorer = WeatherStyleInjectionScore(device)

def load_video_frames(path, n=16):
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    indices = torch.linspace(0, len(vr)-1, n).long()
    frames = vr.get_batch(indices.tolist())
    return frames.permute(0,3,1,2).float() / 255.0

ref     = load_video_frames("ref_videos/sandstorm.mp4").to(device)
base    = load_video_frames("outputs/baseline_no_ref.mp4").to(device)
injected = load_video_frames("outputs/sandstorm_with_clip.mp4").to(device)

print(f"Baseline WSIS:  {scorer.compute(base, ref):.4f}")
print(f"Injected WSIS:  {scorer.compute(injected, ref):.4f}")