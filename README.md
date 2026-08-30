# Rocky AI

**This is a non profit fan project.** Nothing here is sold, monetized or affiliated with anyone. Rocky and the Project Hail Mary universe belong to Andy Weir and the film rights holders. The 3D body is based on the "Project Hale Mary - Rocky" model by orbis_system on Sketchfab. The voice model was trained with Applio for personal fan use only. If you reuse anything from this repository, keep it non commercial and credit the original creators.

A private companion inspired by Rocky from Project Hail Mary. You talk, he listens, thinks, answers with his own voice and reacts with an animated 3D body inside his workshop. Everything runs on your own machine. Nothing goes to the cloud unless you deploy it there yourself.

## What is included

* The full server and web app (chat page and live call page)
* Rocky's rigged 3D body with 12 named animation clips (`model/rocky.glb`)
* Rocky's trained RVC voice (`voice/rocky_private/`, drop it into `Applio/logs/`)
* Cloud scripts that reproduce the whole thing on a rented GPU (`runpod/`)

The AI models themselves are not in the repository. Ollama and faster-whisper download them on first use.

## Features

### Talking with him

* Live call mode: press the microphone once and just talk. Voice activity detection notices when you start and stop speaking and sends each phrase by itself, no buttons.
* Barge in: speak over Rocky and he goes quiet, cancels the rest of his own speech and listens, like interrupting a real person.
* His own voice: Kokoro speaks, then the trained RVC model makes it sound like Rocky. Short replies come as one continuous audio, long ones stream sentence by sentence so he starts talking sooner.
* Captions appear in the exact moment the voice starts, never before.
* Emotions change how he speaks: excited is a little faster, sad a little slower.
* The classic chat page still exists, with push to talk, typing, a camera button, reply audio replay and a clear chat button.
* Hearing that defends itself: hallucinated transcriptions are filtered out, a muted or too quiet microphone triggers a warning instead of invented text, and virtual audio devices like BlackHole are skipped automatically when picking the microphone.

### Seeing you

* A live eye: during a call the camera feeds the vision model in the background every few seconds, so his idea of what is in front of him is always current and answers never wait for image analysis.
* Say something visual ("look at this") and he takes a fresh look on the spot.
* Your own video shows as a small mirror in the corner, like a real video call.

### His body

* Rigged 3D body with 12 animation clips, framed inside his stone workshop with flickering lamps, a workbench, drifting dust and fog.
* Gestures follow the emotion of each reply: jazz hands when happy, a closed fist for no, one finger up for wait, a drooped carapace when sad, full excitement wiggle, and a fist bump when you greet him.
* He is never frozen: idle fidgets, he glances around, and his body reacts to the sound of his own voice.
* Leave him alone for a few minutes and he falls asleep on the workshop floor, then wakes when you speak.
* The waiting states are themed as a light link between Earth and Erid, 16.3 light years apart, with a packet of light traveling the header bar in each direction.

### His mind

* Post story persona: the mission succeeded, he lives on Erid, has a workshop, opinions and boundaries. He never pretends to see without camera data and never breaks character.
* Every reply carries a hidden emotion tag that drives the body and the voice.
* Long term memory: after each exchange, stable facts about you are extracted and saved to a local file, and fed back into every future conversation. He learns your name, your projects, and what you have shown him through the camera. Delete a line in `memory.json` and he forgets it.
* Conversation history is shared between the chat page and the call page.

### For tinkerers

* Everything is local. No accounts, no API keys, no telemetry.
* Developer panel on the call page (press D): what the vision model sees, what Whisper understood with confidence and audio levels, model timings, emotion picked, and every voice activity event.
* All models are swappable through environment variables, and the `runpod/` scripts rebuild the entire thing on a rented GPU in about fifteen minutes.

## Install

1. Install [Ollama](https://ollama.com) and pull a chat model and a vision model from the tables below, for example:

```
ollama pull qwen2.5:3b
ollama pull gemma3:4b
```

2. Install [Applio](https://applio.org) and copy `voice/rocky_private/` into `Applio/logs/`. Start it with `python app.py --port 6969 --server-name 127.0.0.1`.

3. Create the Python environment and run:

```
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python rocky_chat.py
```

Open `http://127.0.0.1:7860` for the chat and `http://127.0.0.1:7860/facetime` for the live call. Pick different models with environment variables: `ROCKY_CHAT_MODEL`, `ROCKY_VISION_MODEL`, `ROCKY_WHISPER`.

## Tested AI models and what each one needs

Every model below was actually run in this project. Latencies are warm measurements.

### Chat (the brain)

| Model | Needs | Reply time | Verdict |
| --- | --- | --- | --- |
| qwen2.5:3b | 8 GB unified memory (Apple Silicon) or 4 GB VRAM | about 1 s | Works well, the default for small machines |
| qwen2.5:14b | 12 GB VRAM or more | 0.5 s | Noticeably smarter persona |
| qwen2.5:32b | 24 GB VRAM or more | 1.1 s | The best conversation quality tested |
| qwen2.5:1.5b | 4 GB memory | 0.5 s | Rejected: breaks character, claims it can see |

### Vision (the eyes)

| Model | Needs | Per frame | Verdict |
| --- | --- | --- | --- |
| gemma3:4b | 8 GB unified memory or 6 GB VRAM | 10 to 20 s | Works, fine descriptions, slow on CPU class machines |
| gemma3:12b | 12 GB VRAM or more | about 3 s | Much better descriptions |
| gemma3:27b | 24 GB VRAM or more | about 2 s | The best eyes tested |

### Hearing (speech to text, faster-whisper)

| Model | Needs | Per phrase | Verdict |
| --- | --- | --- | --- |
| base | any CPU | 1 s | Rejected: hallucinates on quiet audio |
| small | any modern CPU | 3 to 6 s | Good accuracy, the default on CPU |
| large-v3 | 4 GB VRAM | under 1 s | Near perfect, use it whenever a GPU exists |

### Voice (text to speech)

| Piece | Needs | Notes |
| --- | --- | --- |
| Kokoro (am_michael) | CPU or any GPU | fast everywhere |
| Applio RVC (rocky_private) | CPU works, CUDA shines | full spoken reply: 8 to 12 s on an M2 CPU, 1 to 2 s on CUDA |

## Whole machine tiers, measured

| Tier | Machine | Text reply | Full spoken reply |
| --- | --- | --- | --- |
| Minimum | Apple M2, 8 GB | about 1 s | 8 to 12 s |
| Recommended | 24 to 32 GB VRAM GPU | 0.5 to 1.1 s | 1.2 to 2.2 s |
| Cloud reference | RunPod PRO 6000 MIG 48GB at about one dollar per hour | 1.1 s with the 32B brain | 2.2 s |

You also need around 10 GB of disk for the small models, more for the big ones.

## The pipeline

Your voice > Whisper > Qwen with the Rocky persona and long term memory > Kokoro > Applio RVC > browser audio, captions and 3D gestures. Camera frames go to Gemma in a background loop. Developer panel on the call page: press D.

## Credits

* Character and universe: Andy Weir, Project Hail Mary
* Base 3D model: "Project Hale Mary - Rocky" by orbis_system on Sketchfab, rigged and animated for this project
* Voice conversion: Applio and the RVC ecosystem
* Speech: Kokoro TTS and faster-whisper
* Brains: Qwen and Gemma through Ollama
* three.js renders the workshop, MIT licensed files in `vendor/`
