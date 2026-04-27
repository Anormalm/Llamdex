from src.multimodal.task_matrix import datasets as ds
from src.multimodal.task_matrix.runner import TaskSpec


def test_task_spec_prompt_template_style_default_legacy():
    t = TaskSpec(name="finegrained")
    assert t.prompt_template_style == "legacy"


def test_dataset_prompt_tokens_supports_required_datasets():
    dtd = ds._dataset_prompt_tokens("dtd")
    pet = ds._dataset_prompt_tokens("oxford_pet")
    cifar = ds._dataset_prompt_tokens("cifar10")

    assert "DTD schema" in dtd["finegrained"]
    assert "Oxford-IIIT Pet schema" in pet["finegrained"]
    assert "CIFAR-10" in cifar["finegrained"]
    assert "<texture_class>" in dtd["strict_yesno"]
    assert "<breed_name>" in pet["population"]
    assert "<cifar_class>" in cifar["population"]


def test_compact_yesno_template_placeholder_replacement():
    prompt_spec = ds._dataset_prompt_tokens("dtd")
    rendered = prompt_spec["strict_yesno"].replace("<texture_class>", "striped")
    assert "`Yes`" in rendered and "`No`" in rendered
    assert "striped" in rendered
