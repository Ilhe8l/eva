# Changelog

## [0.2.0](https://github.com/Ilhe8l/eva/compare/v0.1.0...v0.2.0) (2026-09-29)


### Features

* an animated pixel-art face for Eva ([5f3a93a](https://github.com/Ilhe8l/eva/commit/5f3a93a28903211ddf2fbbb07891a9a86daebd51))
* Eva keeps a visible plan for multi-step work ([facf78d](https://github.com/Ilhe8l/eva/commit/facf78d03999e391283ed546188358ce476d4fc3))
* report tool results and let Eva choose her expressions ([e2cb0da](https://github.com/Ilhe8l/eva/commit/e2cb0da67b41490a2a1b359ca50262e49821dabd))
* split-screen terminal with Eva's face ([3420eb0](https://github.com/Ilhe8l/eva/commit/3420eb0bffbf403ef2736a4453c8d90b1416ba73))


### Bug Fixes

* an approval never gets stuck behind text already in the input box ([511e092](https://github.com/Ilhe8l/eva/commit/511e0924ed80a861e378c43a5d55482343c6f3d5))
* keep the Activity panel visible on short terminals ([71252d9](https://github.com/Ilhe8l/eva/commit/71252d95e54b1e5b01348d7054e5c4d36f84c269))
* long messages wrap in the input box instead of scrolling sideways ([315d83f](https://github.com/Ilhe8l/eva/commit/315d83f31d84798110553f0b274c39833d4f496b))
* the status line names the reaction Eva's face is playing ([f35595b](https://github.com/Ilhe8l/eva/commit/f35595b24421d6e7050ab987ebaec3ed4860bde3))


### Performance Improvements

* repaint side panels only when their content changes ([e930a16](https://github.com/Ilhe8l/eva/commit/e930a168af7dcfae281e047767e42280b89880bb))


### Documentation

* show the split screen and Eva's face in the README ([f69ea4c](https://github.com/Ilhe8l/eva/commit/f69ea4cdba7f5524860de34ffb0f0dba9adcecd7))

## 0.1.0 (2026-09-28)


### Features

* autonomous mode and 'always allow' approvals ([050e5f3](https://github.com/Ilhe8l/eva/commit/050e5f32d0029645f5c5379f5d1922cf949118c2))
* autonomy with heartbeat, follow-ups and an async terminal ([9a88239](https://github.com/Ilhe8l/eva/commit/9a882397e5668480aa5e9769d2d375ba424d4ca4))
* background tasks, interruptions and fuller speech ([a7ca8a0](https://github.com/Ilhe8l/eva/commit/a7ca8a04fbdccdce7ed73c56f533ca548b979a73))
* configurable step limit, with a summary instead of an error ([fe89a06](https://github.com/Ilhe8l/eva/commit/fe89a06eab988280388a3b13fbaf7bd3d0500341))
* configure Whisper's GPU precision with EVA_WHISPER_COMPUTE_TYPE ([8a4d92a](https://github.com/Ilhe8l/eva/commit/8a4d92a112214c9434c08ce693f4c694d662e075))
* desktop notifications when tasks finish or need approval ([9caf0b5](https://github.com/Ilhe8l/eva/commit/9caf0b5653ad22ecc96261fc80ee4dda0d526478))
* Eva keeps the user posted on long work ([4febccb](https://github.com/Ilhe8l/eva/commit/4febccb438c28d3d1d58a344ef2ac760f5c37c9f))
* eva's personality and docker setup ([ec1e5d7](https://github.com/Ilhe8l/eva/commit/ec1e5d73466ef287bf59d9796406d97e96d4f3be))
* first version with terminal, command approval and journal ([179d295](https://github.com/Ilhe8l/eva/commit/179d2955fb7110aef81e65b2a4a8fd342fce2803))
* gemini support and voice on the gpu ([0f6097a](https://github.com/Ilhe8l/eva/commit/0f6097a97518988f23cad18803325f921ccbd076))
* hands-free mode with speech detection (Silero VAD) ([3527184](https://github.com/Ilhe8l/eva/commit/3527184c3c88282925a9ce481c69a3ce261ddf76))
* let eva propose changes to her own code ([3c5a7e0](https://github.com/Ilhe8l/eva/commit/3c5a7e018ca014136f9af1d279a1f327bdf1c1eb))
* let read-only commands discard errors and sort without approval ([c959169](https://github.com/Ilhe8l/eva/commit/c9591693a69d56b57c42df7b962b308193c30da1))
* messages sent mid-task reach Eva at her next step ([8b8a376](https://github.com/Ilhe8l/eva/commit/8b8a37635a2887f56786e23c8ca89c596bc176a5))
* render Eva's replies as Markdown in the terminal ([39cc4d9](https://github.com/Ilhe8l/eva/commit/39cc4d91337347c2225adea7e58e7f57c0d514be))
* self-extension through skills and reviewed source edits ([72080d4](https://github.com/Ilhe8l/eva/commit/72080d4eab9ae13febb3cfdd0787686e1c3598cf))
* session summary in the journal ([2a0cc21](https://github.com/Ilhe8l/eva/commit/2a0cc214633c37b0db7edda3be9951188bc7f7e9))
* short startup line and a :help command ([317d550](https://github.com/Ilhe8l/eva/commit/317d550c493a078d2f94c019db6b1cd96f6ffca3))
* status speech during turns, .env config ([dcf63a1](https://github.com/Ilhe8l/eva/commit/dcf63a1f8226226a350189e300b13f6f826371b6))
* stream events and speak while working ([00a6c3a](https://github.com/Ilhe8l/eva/commit/00a6c3abc1782cba934e4a7ca5a01a2524338874))
* stream the reply to the terminal as it is written ([db4b343](https://github.com/Ilhe8l/eva/commit/db4b343542aff0a75dcabe7bdf89fd7cd9b8eb3f))


### Bug Fixes

* :record hung waiting for an Enter that was never read ([02fc6c5](https://github.com/Ilhe8l/eva/commit/02fc6c5105bd1c0a7bc50fa6b6705cd7f04c72db))
* explain how to install voice when it is missing ([954a52c](https://github.com/Ilhe8l/eva/commit/954a52cd59501f8b9a7a9ef541cb7bc9867c16a5))
* keep Eva's voice working after a sentence fails ([35d3f1d](https://github.com/Ilhe8l/eva/commit/35d3f1dd06392011fd6b6908654b762d56d3b95f))
* only speak the text eva picks, report rejected actions ([375ba4d](https://github.com/Ilhe8l/eva/commit/375ba4da59e9a5229a5071dd30764d5d689698cb))
* run Whisper on the GPU with CUDA 12 cuBLAS, fall back to CPU ([49990c4](https://github.com/Ilhe8l/eva/commit/49990c4b2187d60a82f9fd9c1e4a9444360e43ac))
* say when the speech model is downloading or loading ([b8fef05](https://github.com/Ilhe8l/eva/commit/b8fef05cf83fb8b8742c4c317b22dc2044b037a1))
* tell the model whether speech is on, keep replies in English ([007483e](https://github.com/Ilhe8l/eva/commit/007483e46caee118adc948bd581a9ee251650b8a))


### Performance Improvements

* run Whisper as int8_float16 on the GPU ([282ea29](https://github.com/Ilhe8l/eva/commit/282ea29447c2a776270b55ca2e3b8e3227236857))
* stream speech by sentence and warm up the voice models ([b214c38](https://github.com/Ilhe8l/eva/commit/b214c38dc5d7e6a052c761497b5ca4e5feb868b4))


### Documentation

* add a demo recording to the README ([608eac4](https://github.com/Ilhe8l/eva/commit/608eac4a331d2de7289cae96111e371071310831))
* add MIT license, security policy and project metadata ([500ec7e](https://github.com/Ilhe8l/eva/commit/500ec7eba1189bad5b72e4b4364a219b94fbb423))
* contributing guide, issue forms and Docker user setup ([feec1d8](https://github.com/Ilhe8l/eva/commit/feec1d81d95b66eebd27f9cb72999c3042867dd6))
* document a local setup for a 6 GB GPU ([5ecd0f9](https://github.com/Ilhe8l/eva/commit/5ecd0f967be779bcc1adb2231bc6a577d7b1d5d4))
* re-record the demo in 16:9 with the GitHub dark theme ([d7d90ef](https://github.com/Ilhe8l/eva/commit/d7d90ef16f6aa78ea6f24cbaab0f0ca2d27d42d6))
* redesign the README ([b088c0e](https://github.com/Ilhe8l/eva/commit/b088c0e01a8e95999bc2a0515ee9bec7456d7a7e))
* rewrite the README around running Eva locally ([6b52c56](https://github.com/Ilhe8l/eva/commit/6b52c56284a0a15d88e4a9f6ce4bdd13cf8a3816))
* use the recommended Qwen3.5 model in .env.example ([4d47f81](https://github.com/Ilhe8l/eva/commit/4d47f81e23ed973551c5fce988401013b382277f))
