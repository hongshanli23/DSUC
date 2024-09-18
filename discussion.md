Hi, 

I wanted to try out universal checkpoint. A typical use case of UC is that cluster size changes and you have to resume the training with different number of GPUs. This also means gradient accumulation steps need to change to ensure consistent global batch size. 

I created a minimal script to test out the correctness of resuming from universal checkpoint with different gradient accumulation steps. However, I found the loss trajectory of the resume run does not track my expectation. 

Here is how I tested it and I will attach the code to repro below. Throughout, I am using ds stage 3

1. Train a GPT like model for 200 steps with 4 GPUs with global batch size 256 and per device batch size of 64 (so gradient accumulation is 1). Save ckpt for 100 step
2. Convert the ZeRO checkpoint at step 100 to an universal checkpoint 
3. Resuming training from the universal checkpoint at step 100 and continue train with 2 GPUs with global bs 256 and per device bs 64 (so gradient accumulation is 2)
