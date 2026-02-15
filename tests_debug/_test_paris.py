import setup_path  # noqa - adds project root to sys.path
from inference.p2p_inference import P2PInferenceEngine

engine = P2PInferenceEngine(
    'P:/Projets/Inference/models/Magistral-Small-2509-Q4_K_M_fragments',
    cache_weights=True
)

engine.generate(
    prompt="dans quel pays est la ville de paris",
    max_tokens=30,
    temperature=0.0
)
