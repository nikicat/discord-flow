Discord-flow - Discord bot for fun and voice communication (development version)
==

Skills
==

 - cities game
 - Akinator
 - AIDungeon (1.x version)
 - DuckDuckGo search
 - ParlAI BST conversations

Available TTS engines
==

 - Google TTS
 - Yandex.SpeechKit

Installation
===

Requirements: python3.9, gcc, gcc-fortran, rust-nightly compilers
Create virtualenv and install packages (I suggest using [direnv](direnv.net) for it):

```
poetry install
```

Set credentials
```
export DIALOGFLOW_PROJECT_ID=<project ID for DialogFlow>
export TOKEN=<Discord bot token>
export YANDEX_API_KEY=<API key from Yandex Cloud management console>
export GOOGLE_APPLICATION_CREDENTIALS=<path to google-application-credentials.json>
```

Then run

```
python -m discordflow
```

and invite bot to your Discord server, it will join first voice channel. Say one of Porcupine wakeup words then ("computer" for example) and ask your question.
