<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#️-description">Description</a></li>
        <li><a href="#-planned-features">Planned Features</a></li>
        <li><a href="#️-built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#-getting-started">Getting Started</a>
      <ul>
        <li><a href="#-installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#-contributing">Contributing</a>
      <ul>
        <li><a href="#-license">License</a></li>
        <li><a href="#-contact">Contact</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# 🧠 About The Project

<p align="center">
  <a href="https://nabster.dev">
    <img src="ui/public/assets/images/screenshot.png" alt="Screenshot" width="100%" height="400" />
  </a>
</p>



<!-- DESCRIPTION -->
### ℹ️ Description

Sawt-AI is an AI-powered application designed to detect and classify Quranic verses from audio inputs. It leverages machine learning techniques to process and analyze audio data for accurate recognition.

- 🎧 Audio Processing: Transcribes recitations from recorded or uploaded audio files.
- 🧠 Machine Learning Models: Combines Whisper transcription, verse matching, and optional imam prediction.
- 📊 Dataset Management: Includes structured datasets for training and evaluation purposes.

---

## 🚀 Planned Features

- 🔍 Enhanced Accuracy: Improve model precision through advanced training techniques.
- 🌐 Web Interface: Develop a user-friendly web interface for easier interaction.
- 📱 Mobile Compatibility: Optimize the application for mobile device usage.
- 🗣️ Real-time Detection: Enable real-time audio analysis and verse recognition.
- 🛠️ Customization Options: Allow users to customize detection parameters and settingsd.

---



### 🏗️ Built With

* [![Python][Python.io]][Python-url]
* [![Docker][Docker.io]][Docker-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
# ✅ Getting Started

This project is now split into two services:

- `ui`: Nuxt application on `http://localhost:3000`
- `api`: FastAPI service on `http://localhost:8000`

### 💻 Installation

```bash
# Clone the repository
git clone https://github.com/nlabrazi/sawt-ai.git
cd sawt-ai

# Start both services
docker compose up --build
```

### ▶️ Usage

1. Open `http://localhost:3000`
2. Press the central microphone once to start recording
3. Recite a Quran passage, then press the same button again to stop and analyze
4. Alternatively, upload an existing audio file
5. The UI sends the completed audio to `POST /recognize` on the API
6. The API returns:
   - Arabic transcription
   - A confirmed Quran passage when the evidence is sufficient
   - An explicitly labelled proposal when two neighbouring passages remain plausible
   - Imam predictions if enabled

French speech, conversations, songs, silence, and uncertain audio are expected
to return no Quran passage instead of a low-confidence guess.
The application does not listen continuously: recording ends only after the
second press or when the 90-second safety limit is reached.

### 🔧 Local Notes

- The API healthcheck is available at `http://localhost:8000/health`
- Supported audio types include `wav`, `mp3`, `m4a`, `ogg`, `webm`
- The uploaded file limit is `12 MB`
- The maximum audio duration expected by the UI is `90 seconds`
- Imam detection depends on the model mounted from `./training`

### 🧪 Tests

Backend API test runner with your `py=/usr/bin/python3` alias:

```bash
py test
```

Backend API tests:

```bash
python3 -m venv api/.venv
api/.venv/bin/pip install -r api/requirements-test.txt
api/.venv/bin/pytest -c api/pytest.ini
```

Frontend unit tests:

```bash
cd ui
npm test
```

Verse detection quality benchmark (text matching only):

```bash
api/.venv/bin/python api/scripts/evaluate_verse_detection.py
```

The versioned corpus is stored in `api/evaluation/verse_detection_corpus.json`.
Add transcriptions observed from real audio to this file before tuning detection
thresholds. This benchmark measures exact passage accuracy, precision, recall,
false positives, and matching latency; it does not measure Whisper accuracy.

End-to-end backend audio smoke benchmark (generated locally, with no downloaded corpus):

```bash
api/.venv/bin/python api/scripts/build_audio_evaluation_corpus.py
docker compose exec api python scripts/evaluate_audio_recognition.py
```

See [`api/evaluation/AUDIO_BENCHMARK.md`](api/evaluation/AUDIO_BENCHMARK.md) for
the private-recitation injection point, consent rules, noisy SNR variants,
offline model setup, quality metrics, and CI-style quality gates.

The current test suite covers:

- FastAPI routes for `recognize`, `feedback`, and `tajwid`
- language screening, transcription policy, passage ranking, and rejection reasons
- reproducible text and audio evaluation metrics with manual release gates
- frontend recording transitions, double-click protection, result/rejection screens, and navigation
- feedback, tajwid loading, confidence rendering, accessibility, and utility parsing

### 🌍 Environment Variables

Example API variables are available in [`api/.env.example`](api/.env.example):

```env
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
WHISPER_MODEL_NAME=turbo
QURAN_VERSETS_PATH=/app/assets/quran_versets.json
TAJWID_DATA_PATH=/app/assets/quran_tajwid.json
TAJWID_BACKUP_URL=https://<project-ref>.supabase.co/storage/v1/object/public/assets/quran_tajwid.json
IMAM_MODEL_PATH=/training/artifacts/models/imam_ecapa_v2/best_model.pt
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_API_KEY=<service_role_or_sb_secret>
SUPABASE_FEEDBACK_TABLE=feedbacks
```

Set `ALLOWED_ORIGINS` with each trusted frontend origin explicitly.
Example for production: `https://sawt-ai.nabster.dev`.
Set `NUXT_PUBLIC_SITE_URL` to the public frontend origin (without a trailing slash) so social sharing metadata uses absolute URLs.
Do not rely on wildcard preview domains when credentials are enabled.
The tajwid loading order is: local snapshot, backup URL, then external API.
`TAJWID_BACKUP_URL` works well with a public JSON file stored in Supabase Storage.
Use the Supabase Project URL, not the Postgres connection string, for `SUPABASE_URL`.
Use a server-side key only for `SUPABASE_API_KEY`, not an `anon` or `sb_publishable` key.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
# 🙌 Contributing

We welcome all contributions! 🛠️ Whether it's fixing a typo, improving documentation, or suggesting a new feature — **every little bit helps**.

To contribute:
1. 🍴 Fork the repo
2. 🔧 Create a feature branch (`git checkout -b feat/my-feature`)
3. 💬 Commit your changes (`git commit -m "feat: add my feature"`)
4. 🚀 Push to your fork (`git push origin feat/my-feature`)
5. 📨 Open a pull request

Thanks a lot for your support! 💙

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
### 📄 License

This project is licensed under the **MIT License** 📜.
You're free to use, modify, and distribute it — just remember to give credit 🤝.

See the full license in [`LICENSE.txt`](https://en.wikipedia.org/wiki/MIT_License) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
### 📬 Contact

- 🛟 [Support and bug reports][issues-url]
- 📧 Configure `NUXT_PUBLIC_CONTACT_EMAIL` with the branded Sawt mailbox before deployment; no personal email is bundled in the application
- 📁 [Project Repository](https://github.com/nlabrazi/sawt-ai)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/nlabrazi/sawt-ai.svg?style=for-the-badge
[contributors-url]: https://github.com/nlabrazi/sawt-ai/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/nlabrazi/sawt-ai.svg?style=for-the-badge
[forks-url]: https://github.com/nlabrazi/sawt-ai/network/members
[stars-shield]: https://img.shields.io/github/stars/nlabrazi/sawt-ai.svg?style=for-the-badge
[stars-url]: https://github.com/nlabrazi/sawt-ai/stargazers
[issues-shield]: https://img.shields.io/github/issues/nlabrazi/sawt-ai.svg?style=for-the-badge
[issues-url]: https://github.com/nlabrazi/sawt-ai/issues
[license-shield]: https://img.shields.io/github/license/nlabrazi/sawt-ai.svg?style=for-the-badge
[license-url]: https://github.com/nlabrazi/sawt-ai/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/nabil-labrazi
[product-screenshot]: app/assets/images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[Rails.js]: https://img.shields.io/badge/rails-%23CC0000.svg?style=for-the-badge&logo=ruby-on-rails&logoColor=white
[Rails-url]: https://rubyonrails.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Ruby.js]: https://img.shields.io/badge/ruby-%23CC342D.svg?style=for-the-badge&logo=ruby&logoColor=white
[Ruby-url]: https://www.ruby-lang.org/en/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
[Javascript.js]: https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E
[Javascript-url]: https://developer.mozilla.org/en-US/docs/Web/JavaScript
[NodeJs.js]: https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white
[NodeJs-url]: https://nodejs.org/en/
[TypeScript.js]: https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white
[TypeScript-url]: https://www.typescriptlang.org/
[RxJS.js]: https://img.shields.io/badge/rxjs-%23B7178C.svg?style=for-the-badge&logo=reactivex&logoColor=white
[RxJS-url]: https://rxjs.dev/
[NestJs.io]: https://img.shields.io/badge/nestjs-E0234E?style=for-the-badge&logo=nestjs&logoColor=white
[NestJs-url]: https://nestjs.com/
[Prisma.io]: https://img.shields.io/badge/Prisma-3982CE?style=for-the-badge&logo=Prisma&logoColor=white
[Prisma-url]: https://www.prisma.io/
[Python.io]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[Railway.io]: https://img.shields.io/badge/Railway-000000?style=for-the-badge&logo=railway&logoColor=white
[Railway-url]: https://railway.app/
[Docker.io]: https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
[PostgreSQL.js]: https://img.shields.io/badge/postgresql-316192?style=for-the-badge&logo=postgresql&logoColor=white
[PostgreSQL-url]: https://www.postgresql.org/
[TailwindCSS.js]: https://img.shields.io/badge/tailwindcss-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white
[TailwindCSS-url]: https://tailwindcss.com/
[Stimulus.js]: https://img.shields.io/badge/stimulus-0a0a0a?style=for-the-badge&logo=stimulus&logoColor=white
[Stimulus-url]: https://stimulus.hotwired.dev/
