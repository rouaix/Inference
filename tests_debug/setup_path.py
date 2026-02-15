"""
setup_path.py
=============
Ajoute la racine du projet a sys.path.

Importer en premier dans chaque script de tests_debug/ pour que les
packages du projet (inference/, fragments/, distribution/, dequantize/)
soient trouvables quel que soit le repertoire de lancement.

Usage
-----
    import setup_path  # noqa — doit etre le premier import
    from inference.p2p_inference import P2PInferenceEngine
"""
import sys
from pathlib import Path

_root = Path(__file__).parent.parent.resolve()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
