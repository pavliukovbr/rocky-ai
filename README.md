# Rocky AI

**This is a non profit fan project.** Nothing here is sold, monetized or affiliated with anyone. Rocky and the Project Hail Mary universe belong to Andy Weir and the film rights holders. The 3D body is based on the "Project Hale Mary - Rocky" model by orbis_system on Sketchfab. The voice model was trained with Applio for personal fan use only. If you reuse anything from this repository, keep it non commercial and credit the original creators.

A private companion inspired by Rocky from Project Hail Mary. You talk, he listens, thinks, answers with his own voice and reacts with an animated 3D body inside his workshop. Everything runs on your own machine. Nothing goes to the cloud unless you deploy it there yourself.

## What is included

* The full server and web app (chat page and live call page)
* Rocky's rigged 3D body with 12 named animation clips (`model/rocky.glb`)
* Rocky's trained RVC voice (`voice/rocky_private/`, drop it into `Applio/logs/`)
* Cloud scripts that reproduce the whole thing on a rented GPU (`runpod/`)

The AI models themselves are not in the repository. Ollama and faster-whisper download them on first use.

## What he does

* Real voice conversation. On a call the microphone stays open, he detects when you start and stop talking and answers by himself. Speak over him and he goes quiet and listens, like a real call.
* His own voice. Kokoro speaks, then the RVC model makes it sound like Rocky.
* He sees you. The camera feeds a local vision model in the background, so he always knows what is in front of him.
* He feels. Every reply carries an emotion that drives his gestures: jazz hands when happy, a closed fist for no, a drooped carapace when sad, a fist bump when you greet him.
* He remembers. Facts about you are stored in a local file and fed back into every future conversation.
* He sleeps. Leave him alone for a few minutes and he curls up on the workshop floor.

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
