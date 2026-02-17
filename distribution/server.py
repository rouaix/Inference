"""
FastAPI Server for P2P Inference Network

Each node runs this server to expose its local fragments and perform layer-wise inference.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Local imports
from distribution.local import LocalFragmentLoader
from inference.p2p_inference import P2PInferenceEngine, LlamaLayer, rms_norm

app = FastAPI(title="P2P Inference Node", version="0.1.0")

class NodeState:
    def __init__(self):
        self.fragments_dir: Optional[str] = None
        self.layers: Optional[List[int]] = None
        self.engine: Optional[P2PInferenceEngine] = None
        self.loader: Optional[LocalFragmentLoader] = None
        self.model_name: Optional[str] = None
        self.node_id: Optional[str] = None
        self.session_cache: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

state = NodeState()

@app.on_event("startup")
def startup_event():
    """Initialize node state when server starts."""
    global state
    
    if not state.fragments_dir or not state.layers:
        raise RuntimeError("Node not properly initialized. Use initialize_node() first.")
    
    # Load fragments
    state.loader = LocalFragmentLoader(state.fragments_dir, verbose=True)
    
    # Initialize engine
    state.engine = P2PInferenceEngine(state.fragments_dir, verbose=True)
    
    # Extract model name from manifest
    manifest_path = Path(state.fragments_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        state.model_name = manifest.get("model_name", "unknown")
    
    # Generate node ID (simple for now)
    state.node_id = f"node-{state.layers[0]}-{state.layers[-1]}"
    
    print(f"Node {state.node_id} initialized with layers {state.layers}")

@app.get("/status")
def get_status():
    """Return node status and capabilities."""
    if not state.engine:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    return {
        "node_id": state.node_id,
        "layers": state.layers,
        "model": state.model_name,
        "ready": True,
        "fragments_dir": state.fragments_dir
    }

@app.get("/manifest")
def get_manifest():
    """Return manifest metadata (without weight bytes)."""
    if not state.fragments_dir:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    manifest_path = Path(state.fragments_dir) / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Remove fragment file paths to keep it clean
    if "fragments" in manifest:
        for frag in manifest["fragments"]:
            frag.pop("fragment_id", None)
    
    return manifest

class ExecuteLayerRequest:
    def __init__(self, hidden_state: List[List[float]], pos: int = 0,
                 cache_k: Optional[List[List[List[float]]]] = None,
                 cache_v: Optional[List[List[List[float]]]] = None):
        self.hidden_state = np.array(hidden_state, dtype=np.float32)
        self.pos = pos
        self.cache_k = np.array(cache_k, dtype=np.float32) if cache_k else None
        self.cache_v = np.array(cache_v, dtype=np.float32) if cache_v else None

class ExecuteLayerResponse:
    def __init__(self, output: np.ndarray, new_k: np.ndarray, new_v: np.ndarray):
        self.output = output
        self.new_k = new_k
        self.new_v = new_v
    
    def to_dict(self):
        return {
            "output": self.output.tolist(),
            "new_k": self.new_k.tolist(),
            "new_v": self.new_v.tolist()
        }

    def _serialize_array(self, arr: np.ndarray):
        """
        Serialize numpy array for binary transmission.
        Uses hex encoding for JSON compatibility.
        """
        if arr is None:
            return None
        
        # Use hex encoding for binary data to ensure JSON compatibility
        hex_data = arr.tobytes().hex()
        
        return {
            "__binary__": True,
            "data": hex_data,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype)
        }

@app.post("/execute_layer")
async def execute_layer(request: Request):
    """Execute a single transformer layer and return output + new KV cache."""
    global state
    
    if not state.engine or not state.loader:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    try:
        # Parse request
        data = await request.json()
        
        # Deserialize arrays (support both JSON and binary formats)
        def deserialize_array(data):
            if isinstance(data, dict):
                if data.get("__binary_zstd__"):
                    import base64
                    try:
                        import zstandard as zstd
                        compressed_data = base64.b64decode(data["data"])
                        decompressed = zstd.ZstdDecompressor().decompress(compressed_data)
                        return np.frombuffer(decompressed, dtype=data["dtype"]).reshape(data["shape"])
                    except ImportError:
                        raise HTTPException(
                            status_code=415,
                            detail="zstandard requis pour les données compressées. "
                                   "Installer avec: pip install zstandard"
                        )
                elif data.get("__binary__"):
                    import binascii
                    bytes_data = binascii.unhexlify(data["data"])
                    return np.frombuffer(bytes_data, dtype=data["dtype"]).reshape(data["shape"])
            elif isinstance(data, list):
                return np.array(data, dtype=np.float32)
            return None
        
        hidden_state = deserialize_array(data["hidden_state"])
        pos = data.get("pos", 0)
        session_id = data.get("session_id")
        layer_idx = data.get("layer_idx")
        
        # Retrieve KV cache from session if available
        cache_k, cache_v = None, None
        if session_id and session_id in state.session_cache and layer_idx in state.session_cache[session_id]:
            cache_k, cache_v = state.session_cache[session_id][layer_idx]
        else:
            cache_k = deserialize_array(data.get("cache_k"))
            cache_v = deserialize_array(data.get("cache_v"))
        
        if layer_idx is None:
            raise HTTPException(status_code=400, detail="layer_idx is required")
        
        if layer_idx not in state.layers:
            raise HTTPException(
                status_code=400,
                detail=f"Layer {layer_idx} not available on this node (has {state.layers})"
            )
        
        # Execute layer
        layer = LlamaLayer(state.engine, layer_idx)
        output, new_k, new_v = layer.forward(
            hidden_state, 
            state.engine.freqs_cis, 
            cache_k, 
            cache_v, 
            start_pos=pos
        )
        
        # Store KV cache in session if session_id provided
        if session_id:
            if session_id not in state.session_cache:
                state.session_cache[session_id] = {}
            state.session_cache[session_id][layer_idx] = (new_k, new_v)
        
        # Return response with binary serialization
        response = ExecuteLayerResponse(output, new_k, new_v)
        
        # Check if client supports binary format by looking at request headers
        accept_binary = request.headers.get("Accept", "").lower().find("application/octet-stream") >= 0
        
        if accept_binary:
            # Return binary format for better performance
            response_dict = {
                "output": self._serialize_array(output),
                "new_k": self._serialize_array(new_k),
                "new_v": self._serialize_array(new_v)
            }
        else:
            # Fall back to JSON format
            response_dict = response.to_dict()
        
        return JSONResponse(content=response_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free its KV cache."""
    global state
    if session_id in state.session_cache:
        del state.session_cache[session_id]
        return {"status": "success", "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

def initialize_node(fragments_dir: str, layers: str, port: int):
    """Initialize node state before starting the server."""
    global state
    
    state.fragments_dir = fragments_dir
    
    # Parse layers range
    if "-" in layers:
        start, end = layers.split("-")
        state.layers = list(range(int(start), int(end) + 1))
    else:
        state.layers = [int(layers)]
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2P Inference Node Server")
    parser.add_argument("fragments_dir", help="Directory containing fragment files")
    parser.add_argument("--layers", required=True, 
                       help="Layer range (e.g., 0-19 or 20)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to listen on")
    
    args = parser.parse_args()
    initialize_node(args.fragments_dir, args.layers, args.port)