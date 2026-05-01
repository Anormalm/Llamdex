from src.multimodal.task_matrix import datasets as ds
from src.multimodal.task_matrix.runner import TaskSpec
from src.multimodal.task_matrix.datasets import ManifestImageDataset, VQASubsetDataset, VQASubsetSpec, _build_codebook


def test_task_spec_prompt_template_style_default_legacy():
    t = TaskSpec(name="finegrained")
    assert t.prompt_template_style == "legacy"


def test_dataset_prompt_tokens_supports_required_datasets():
    dtd = ds._dataset_prompt_tokens("dtd")
    pet = ds._dataset_prompt_tokens("oxford_pet")
    cifar = ds._dataset_prompt_tokens("cifar10")
    eurosat = ds._dataset_prompt_tokens("eurosat")
    food = ds._dataset_prompt_tokens("food101")
    resisc = ds._dataset_prompt_tokens("resisc45")
    aircraft = ds._dataset_prompt_tokens("fgvc_aircraft")

    assert "DTD schema" in dtd["finegrained"]
    assert "Oxford-IIIT Pet schema" in pet["finegrained"]
    assert "CIFAR-10" in cifar["finegrained"]
    assert "EuroSAT" in eurosat["finegrained"]
    assert "Food-101" in food["finegrained"]
    assert "RESISC45" in resisc["finegrained"]
    assert "FGVC-Aircraft" in aircraft["finegrained"]
    assert "<texture_class>" in dtd["strict_yesno"]
    assert "<breed_name>" in pet["population"]
    assert "<cifar_class>" in cifar["population"]
    assert "<land_cover_class>" in eurosat["strict_yesno"]
    assert "<dish_class>" in food["population"]
    assert "<scene_class>" in resisc["strict_yesno"]
    assert "<aircraft_variant>" in aircraft["population"]


def test_compact_yesno_template_placeholder_replacement():
    prompt_spec = ds._dataset_prompt_tokens("dtd")
    rendered = prompt_spec["strict_yesno"].replace("<texture_class>", "striped")
    assert "`Yes`" in rendered and "`No`" in rendered
    assert "striped" in rendered


def test_build_codebook_supports_large_label_spaces():
    class _Tokenizer:
        def __init__(self):
            self.ids = {}

        def encode(self, text, add_special_tokens=False):
            token = str(text).strip()
            self.ids.setdefault(token, len(self.ids) + 100)
            return [self.ids[token]]

    codes, code_to_tid = _build_codebook(_Tokenizer(), 101)
    assert len(codes) == 101
    assert len(code_to_tid) == 101
    assert len(set(code_to_tid.values())) == 101


def test_manifest_image_dataset_loads_relative_paths(tmp_path):
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_dir / "a.png")
    manifest = tmp_path / "train.csv"
    manifest.write_text("image_path,label\nimages/a.png,pneumonia\n", encoding="utf-8")

    dset = ManifestImageDataset(str(manifest), transform=None)
    image, label = dset[0]
    assert image.size == (8, 8)
    assert label == 0
    assert dset.classes == ["pneumonia"]


def test_local_vqa_dataset_supports_second_image_field(tmp_path):
    from PIL import Image

    class _Tokenizer:
        pad_token_id = 0

        def __init__(self):
            self.ids = {}

        def encode(self, text, add_special_tokens=False):
            token = str(text).strip()
            self.ids.setdefault(token, len(self.ids) + 100)
            return [self.ids[token]]

        def apply_chat_template(self, msgs, return_tensors="pt", tokenize=False):
            import torch

            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(tmp_path / "prior.png")
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(tmp_path / "current.png")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        '{"image_path":"prior.png","second_image_path":"current.png","question":"Changed?","answer":"yes"}\n',
        encoding="utf-8",
    )

    dset = VQASubsetDataset(
        tokenizer=_Tokenizer(),
        spec=VQASubsetSpec(
            hf_dataset_name=f"local:{manifest}",
            image_field="image_path",
            second_image_field="second_image_path",
            question_field="question",
            answer_field="answer",
        ),
    )
    row = dset[0]
    assert row["images"].shape == (3, 224, 224)
    assert row["task_name"] == "vqa"
