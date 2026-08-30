# Rocky AI

A private companion inspired by Rocky from Project Hail Mary. You talk, he listens, thinks, answers with his own voice and reacts with an animated 3D body inside his workshop. Everything runs on your own machine. Nothing goes to the cloud.

## What he does

* Real voice conversation. The microphone stays open during a call, he detects when you start and stop talking and answers by himself.
* His own voice. A base TTS voice is converted through an RVC model so he actually sounds like Rocky.
* He sees you. During a call the camera feeds a local vision model in the background, so he always knows what is in front of him and can comment on it.
* He feels. Every reply carries an emotion that drives his 3D gestures: jazz hands when happy, a closed fist for no, a drooped carapace when sad, a fist bump when you greet him.
* He remembers. Facts about you are extracted after each talk and stored in a local file, then fed back into every future conversation.
* He sleeps. Leave him alone for a few minutes and he curls up on the workshop floor until you speak again.

## The pipeline

Your voice > Whisper (speech to text) > Qwen 2.5 3B with the Rocky persona and long term memory > Kokoro (base speech) > Applio RVC (Rocky's voice) > browser audio, captions and 3D gestures. Camera frames go to Gemma 3 4B in a background loop. All of it is local.

## Install

1. Install [Ollama](https://ollama.com) and pull the two brains:

```
ollama pull qwen2.5:3b
ollama pull gemma3:4b
```

2. Install [Applio](https://applio.org) and place your trained RVC voice model in `Applio/logs/<name>/`. Start it with `python app.py --port 6969 --server-name 127.0.0.1`.

3. Create the Python environment for the server:

```
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

4. Point the server at your files with environment variables if your paths differ from the defaults:

```
export ROCKY_APPLIO_DIR="$HOME/Desktop/Applio"
export ROCKY_RVC_NAME="rocky_private"
export ROCKY_RVC_PTH="rocky_private_200e_9400s.pth"
export ROCKY_MODEL_GLB="$HOME/path/to/Rocky_realtime.glb"
```

5. Run it:

```
python rocky_chat.py
```

Open `http://127.0.0.1:7860` for the chat and `http://127.0.0.1:7860/facetime` for the live call.

## How to use

* Chat page: press the round talk button and speak, or type. The camera button attaches a frame to your next message. Settings live behind the gear.
* Call page: press the microphone once and just talk. He hears you continuously, watches you through the camera preview in the corner, answers out loud and acts it out. The red button hangs up.
* Press D or the DEV button on the call page for the developer panel: what the vision model is seeing, what Whisper understood from your voice with confidence and audio levels, model timings and the voice activity detector events.
* He learns about you over time. His notes live in `memory.json` next to the server. Delete a line and he forgets it.

## Hardware and what to expect

Measured on real hardware, warm models, short replies:

| Tier | Machine | Text reply | First spoken words | Vision refresh |
| --- | --- | --- | --- | --- |
| Minimum | Apple M1 or M2, 8 GB | about 1 s | 8 to 12 s | 10 to 20 s per frame |
| Recommended | RTX 2070 Super class, 8 GB VRAM | under 1 s | 2 to 3 s | 2 to 3 s per frame |
| Good | RTX 3080 or better, or M3 Pro 18 GB | under 1 s | about 1 s | about 1 s per frame |

On the minimum tier the models take turns in memory, so the first answer after a camera look can be a few seconds slower. From the recommended tier up everything stays loaded and the conversation flows.

You also need around 10 GB of disk for the models.

## Bring your own Rocky

The 3D model, textures and the trained voice are not part of this repository. This is a personal fan project and character assets are not distributed.

* Voice: train an RVC model in Applio from your own clips and point the environment variables at it.
* Body: any rigged GLB works. Gestures map to animation clips named `idle_loop`, `attention`, `yes_jazz_hands`, `no_fist`, `wait_one_finger`, `sad`, `excited`, `fist_bump`, `sleep_full`, `sleep_loop`, `scamper_loop`. Missing clips are simply skipped. Put the file at the path set in `ROCKY_MODEL_GLB`.

The three.js files in `vendor/` are from the three.js project, MIT licensed.
