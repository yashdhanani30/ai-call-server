from piper import PiperVoice

voice = PiperVoice.load("piper_models/en_US-amy-medium.onnx")

gen = voice.synthesize("Testing Piper chunk structure")

first_chunk = next(gen)

print("TYPE:", type(first_chunk))
print("DIR:", dir(first_chunk))
print("\nATTRIBUTES WITH NON-NONE VALUES:")
for attr in dir(first_chunk):
    if not attr.startswith("_"):
        try:
            value = getattr(first_chunk, attr)
            if value is not None:
                print(attr, "=", type(value))
        except:
            pass
