import torch
import torch.nn.functional as F
import clip

class WeatherStyleInjectionScore:
    def __init__(self, device):
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def compute(self, video_frames, ref_frames):
        # video_frames: [N, C, H, W] tensor, values [0,1]
        # ref_frames:   [M, C, H, W] tensor, values [0,1]
        # returns scalar cosine similarity

        def get_features(frames):
            # resize to 224, normalize for CLIP
            frames = F.interpolate(frames, size=(224, 224), mode='bilinear')
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(self.device)
            std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(self.device)
            frames = (frames - mean) / std
            feats = self.model.encode_image(frames)
            return feats.mean(dim=0)  # average across frames

        f_gen = get_features(video_frames)
        f_ref = get_features(ref_frames)
        score = F.cosine_similarity(f_gen.unsqueeze(0), f_ref.unsqueeze(0))
        return score.item()