from piper import PiperVoice

print("Loading model...")
voice = PiperVoice.load("piper_models/en_US-amy-medium.onnx")

print("Synthesizing audio...")
gen = voice.synthesize("Hello! This is working perfectly.")

output_file = "output.wav"

with open(output_file, "wb") as f:
    for chunk in gen:
        f.write(chunk.audio_int16_bytes)

print("✔ Done! Saved:", output_file)

