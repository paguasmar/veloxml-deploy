import tempfile
from pathlib import Path
from veloxml.config import VeloxConfig

def test_default_config():
    config = VeloxConfig()
    assert config.name == "my-model-service"
    assert config.compute.cloud == "aws"
    assert config.compute.region == "us-east-1"
    assert config.service.port == 8000

def test_save_and_load_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "veloxml.yaml"
        cfg = VeloxConfig(name="custom-model")
        cfg.compute.accelerator = "T4:1"
        cfg.save(config_path)

        loaded = VeloxConfig.load_or_default(config_path)
        assert loaded.name == "custom-model"
        assert loaded.compute.accelerator == "T4:1"
