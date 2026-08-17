import os, glob
from PIL import Image
src = "/tmp/wan_atlas/decoded"
dst = "/tmp/wan_atlas/decoded_small"
n = 0
for p in sorted(glob.glob(os.path.join(src, "*", "*.png"))):
    arm = os.path.basename(os.path.dirname(p))
    out_dir = os.path.join(dst, arm)
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(p).convert("RGB")
    im.thumbnail((512, 512))
    im.save(os.path.join(out_dir, os.path.basename(p).replace(".png", ".jpg")),
            "JPEG", quality=82)
    n += 1
print("wrote", n, "files")
