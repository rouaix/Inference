import gguf
import sys

def inspect_gguf(path):
    print(f"Inspecting {path}...")
    reader = gguf.GGUFReader(path)

    print(f"Header fields: {len(reader.fields)}")
    print(f"Tensors: {len(reader.tensors)}")

    if len(reader.tensors) > 0:
        t = reader.tensors[0]
        print(f"\n--- Tensor 0: {t.name} ---")
        print(f"Attributes: {dir(t)}")
        print(f"Data Offset: {getattr(t, 'data_offset', 'N/A')}")
        # 'offset' represents the offset of the tensor data relative to the start of the data section?
        # Or absolute?
        print(f"Field 'offset': {getattr(t, 'offset', 'N/A')}")

        # Check alignment guess
        # usually header size + tensor info = data_offset

    # Let's verify where the first tensor starts in the file
    pass

if __name__ == "__main__":
    inspect_gguf(sys.argv[1])
