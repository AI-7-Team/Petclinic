import os
import json
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class LesionDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs  # 여러 디렉토리 경로
        self.transform = transform
        self.img_files = []
        self.class_to_idx = {}

        # 모든 이미지 파일 수집
        for root_dir in self.root_dirs:
            print(root_dir)
            img_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".png") or f.endswith(".jpg")])
            self.img_files.extend([os.path.join(root_dir, f) for f in img_files])

        # 전체 라벨 목록 수집 → 숫자 ID 매핑
        labels = set()
        for img_file in tqdm(self.img_files):
            json_path = img_file.rsplit(".", 1)[0] + ".json"
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                lesion_code = data["metaData"]["lesions"]
                labels.add(lesion_code)

        self.class_to_idx = {name: i for i, name in enumerate(sorted(labels))}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        json_path = img_path.rsplit(".", 1)[0] + ".json"

        img = Image.open(img_path).convert("RGB")

        with open(json_path, "r", encoding="utf-8") as jf:
            meta = json.load(jf)
        lesion_code = meta["metaData"]["lesions"]
        label = self.class_to_idx[lesion_code]

        if self.transform:
            img = self.transform(img)

        return img, label