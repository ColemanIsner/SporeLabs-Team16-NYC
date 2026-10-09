# Spore demo video: voiceover script

Edit the lines below, then rebuild with `python video/make_video.py`.
Each `## step` is one screen of the Story UI (http://localhost:5173), in the UI's order.
The screen stays up as long as its lines take to read, plus a short pause.
Each sentence becomes one caption (long ones are split). Write names normally (YOLO, VSS, W&B);
video/tts.py handles pronunciation.

## intro
This is Spore, by SporeLabs. Video AI fails quietly on the conditions your cameras rarely see. Spore finds where your video AI is weak, grows the data to cover it, and tests your models on it, in a loop.

## inventory
First, it looks through everything you have. Spore reads all six hundred twelve real clips indexed in VAST's video search, each with its Cosmos Reason caption.

## missing
Then it finds the weak spots. These are live VSS searches against the highway cameras. Thirty clips in clear daylight. Zero at night. Zero in heavy rain. Zero in fog. Zero in snow. The detector on these cameras has never been tested in any of it.

## matters
An LLM on W&B Inference ranks which gaps matter, and every call is traced in Weave. We'll follow dense fog, where the detector breaks hardest. Glare, night, night rain and snow each get their own loop.

## grow
Next, it grows the missing weather onto a real clip. A physics weather layer adds fog, rain and snow, and NVIDIA Cosmos Transfer turns day into night. The cars stay where they were, so we already know the right answer.

## test
Then it tests the models inside the VSS pipeline. On this foggy clip, the Cosmos Reason caption model still counts twelve vehicles. The YOLO detector misses six in ten cars. Same clip, same pipeline, and the two models disagree.

## blind
Across eleven clips and three severities, the pattern is clear. In light fog, YOLO finds only four in ten cars a person can still see. In heavy fog or heavy rain, it finds one. The whole loop is traced in W&B Weave.

## train
And every frame Spore grows comes already labeled. The weather layer never moves a car, so the twenty-three boxes YOLO found on the clear-day frame line up exactly on the fog, rain and snow versions. That's nine hundred twenty-eight weather frames and nearly twelve thousand vehicle boxes for the Fix step, and not one was drawn by hand.

## again
Then Spore picks the next weak spot. Low sun glare held up: YOLO still finds eight in ten. Next up is the highway at night. Find where your video AI is weak, grow the data to fix it, and go again.
