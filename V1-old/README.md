# Ternary TinyStories (1.58-bit Language Model)

DISCLAIMER: This README was written by AI and is a Work in Progress (WIP).

This is the current state of the project experimenting with ternary and QAT training of small language models. It is a custom implementation of a BitNet-inspired LLM trained on the TinyStories dataset.

The goal is a model that doesn't just "think" in decimals, but in -1, 0, and 1.
The Vibe

Most LLMs are heavy, power-hungry, and obsessed with 16-bit or 32-bit precision. This is an experiment in the Efficiency Frontier. By using 1.58-bit (ternary) weights, the goal is to move toward a future where inference is just a series of additions and subtractions instead of expensive multiplications.
Current Features

    BitNet b1.58 Logic: Custom TernaryLinear layers using a Straight-Through Estimator (STE).

    Stabilized Training: Activation normalization to stop the math from exploding into NaN values.

    Tiny Architecture: Optimized for local training on consumer hardware like an RTX 2080 Super running Pop!_OS.

    Custom Tokenizer: A tight Byte-Level BPE vocab designed to keep the model fast and focused.

Tech Stack

    Framework: PyTorch + Hugging Face Transformers.

    Dataset: TinyStories (synthetic stories for small-scale brain development).

    Quantization: Weight centering and gamma-scaling to maintain intelligence at low bit-widths.

    Hardware Vision: Designed to eventually leave the GPU behind and run on FPGAs like the Tang Nano 20k and the NOMAD open-source laptop.

How to use it

    Train: Run main.py and watch the loss.

    Chat: Use chat.py to talk to whatever checkpoint is available.

    Snap: Use the snap.py utility to freeze the weights into their final ternary form for publication or deployment. (TODO)

Future Roadmap

    90%+ Ternary: Moving the Embeddings and LM Head into ternary space for pure bit-efficiency.

    Weight Packing: Shrinking the model files  by packing ternary values into 2-bit integers.

    Hardware Deployment: Running this "Story AI" at massive FPS on FPGA logic cells.

Credits

Inspired by the BitNet research and the desire to run AI on things that shouldn't be able to run AI.
Final Checkpoint Note

Additionally, the checkpoint files can be tested with chat.py but they are still pretty big because they are in fp32 and store other training data aswell.