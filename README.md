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
* [![Playwright][Playwright.io]][Playwright-url]

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
api/.venv/bin/pytest -c api/pytest.ini api/tests
```

Frontend unit tests:

```bash
cd ui
npm test
```

Frontend End-to-End (E2E) tests with Playwright:

```bash
cd ui
npm run test:e2e
```

Run E2E tests in interactive UI mode or view HTML reports:

```bash
npm run test:e2e:ui
npm run test:e2e:report
```

Run E2E tests with the official Playwright Docker image:

```bash
docker run --rm --ipc=host -e CI=true -w /workspace/ui \
  -v $(pwd):/workspace \
  mcr.microsoft.com/playwright:v1.63.0-noble \
  bash -c "npm ci && npm run test:e2e"
```

Verse detection quality benchmark (text matching only):

```bash
api/.venv/bin/python api/scripts/evaluate_verse_detection.py
```

The versioned corpus is stored in `api/evaluation/verse_detection_corpus.json`.
Add transcriptions observed from real audio to this file before tuning detection
thresholds. This benchmark measures exact passage accuracy, precision, recall,
false positives, and matching latency; it does not measure Whisper accuracy.

Hadith search is available in the **Hadiths** mode of the Nuxt interface (Beta).
Type a subject in French or press the microphone beside the search bar and say,
for example, “Trouve-moi le ou les hadiths qui parlent du mariage”. Press the stop
button to transcribe and search. The recording also stops automatically after
30 seconds. The recognized sentence appears in the search bar; you can correct
it and search again. Open one of up to three proposals to read the Arabic text,
translation, and available source details.
The landing screen links to Quran and Hadith search, with a separate FAQ.
The Sawt AI logo returns home and resets searches. Leaving Hadith mode clears
the query and results. Pending searches are cancelled on exit; Quran recording
prevents switching modes until the recording is finished. Leaving Hadith mode
stops its microphone, cancels pending transcription/search and discards late
responses. Texts come from HadeethEnc and are never generated.

Hadith voice search requires microphone permission and HTTPS (or localhost).
Permission denial, unsupported browsers, unreadable audio and transcription
errors leave typed search available. No continuous listening or spoken answer
is involved: results use the existing Hadith cards.

The UI sends the recorded file as multipart field `file` to
`POST /hadith/transcribe`. The API returns the original French transcription:

```json
{ "query": "Trouve-moi les hadiths qui parlent du mariage" }
```

The UI then calls `POST /hadith/search` with that query and `limit: 3`. Recognized
request prefixes are removed inside the existing search service, preserving the
subject and negations. Voice input uses the same keyword/semantic policy as
text input; it does not guarantee three relevant results or search exhaustiveness.

The transcription endpoint reuses the loaded Whisper model with `language="fr"`
and voice activity detection. Audio upload validation and the concurrency budget
(`MAX_CONCURRENT_INFERENCES`, default 2) are shared with Quran recognition.
The file limit is 12 MB. The server accepts up to 31 seconds, allowing one second
for the browser's asynchronous stop and final codec frame around the UI's
30-second recording limit. Temporary audio is deleted on success and failure;
transcribed text is not logged by the transcription service.

| Status | Meaning |
| --- | --- |
| `400` | Empty audio file |
| `413` | File size or duration limit exceeded |
| `415` | Unsupported signature or undecodable audio |
| `422` | Missing file, no usable spoken query, or query longer than 300 characters |
| `503` | Transcription model unavailable or inference failed |

Restart the API after updating the backend code to register the new endpoint:

```bash
docker compose restart api
```

API tests mock Whisper; browser voice tests use a real MediaRecorder with a
synthetic audio stream and mocked transcription/source responses. Before release,
check a real French microphone recording against the running API, for example
requests about marriage, a request with a negation, silence, and a correction of
the recognized sentence. These automated tests verify the workflow and error
handling, rather than measuring speech recognition or religious relevance.

Build the current E5-base / multi_context index before the first search:

```bash
docker compose exec api python scripts/build_hadith_index.py
```

The index and metadata are local artifacts ignored by Git. Restart the API if it
has already loaded an older index. Results require Internet access to HadeethEnc.
Short subjects (up to three words after removing a recognized search prefix) use
keyword retrieval across French titles, texts, and explanations. Every returned
source must contain all keywords, with simple singular/plural variants. Missing
keywords return an empty list: `couronne` and `hadith couronne` no longer return
unrelated neighbours. Accents matter (`couronne` does not match `couronné`).
Longer phrases and negations retain semantic retrieval. The UI labels the search
method; semantic neighbours may not answer the query. The provisional retrieval
benchmark does not establish religious accuracy.

Upgrade an existing index from its original cached source records without
re-encoding, then restart the API:

```bash
docker compose exec api python scripts/build_hadith_index.py --upgrade-search-documents
docker compose restart api
```

The upgrade verifies the exact source checksum and preserves a metadata backup.
If the cache is missing or changed, rebuild the index instead. New builds include
the source text metadata automatically.

In production, the API image or persistent volume must contain both
`hadith_index.npz` and `hadith_index_meta.json` at the paths configured by
`HADITH_INDEX_PATH` and `HADITH_INDEX_META_PATH`. A fresh Git checkout does not
include them, and `.cache/` is excluded from Docker images. Build or copy the
matching pair before deployment; keyword search requires metadata schema 2 with
`search_documents`. The metadata upgrade requires the original source cache.
API error logs include `errorCauses` to distinguish missing files, incompatible
metadata, embedding model failures, and HadeethEnc connection errors behind a 503.

For the VPS layout with `/srv/apps/sawt-ai/repo` as the Git checkout, keep the
index files in `/srv/apps/sawt-ai/data/hadith`, outside the checkout. In the
existing `/srv/apps/sawt-ai/docker-compose.yml`, add these entries to the
`sawt-api` service, preserving its other settings:

```yaml
    environment:
      HADITH_INDEX_PATH: /app/data/hadith/hadith_index.npz
      HADITH_INDEX_META_PATH: /app/data/hadith/hadith_index_meta.json
    volumes:
      - type: bind
        source: ./data/hadith
        target: /app/data/hadith
        read_only: true
        bind:
          create_host_path: false
```

This mounts the Hadith directory read-only at `/app/data/hadith`. The two path
variables in `environment` override the values in `repo/api/.env`.

Create `data/hadith` on the VPS and transfer both validated files directly into
it before applying this configuration. The mount requires the directory to
exist; it does not create an empty directory silently. Run `docker compose
config --quiet` from `/srv/apps/sawt-ai`, then `docker compose up -d --no-build
sawt-api` to apply the mount to the current image. A simple container restart
does not apply changed mounts or environment variables. Subsequent deployments
can keep using `deploy.sh`: replacing containers and pruning images, build
cache, or stopped containers does not delete this host directory.

After recreating the API, validate the configured artifacts:

```bash
docker exec -i sawt-api python - <<'PY'
from app.core.hadith_config import HadithConfig
from app.services.hadith_index import load_index, load_search_documents

config = HadithConfig.from_env()
load_index(config)
print(len(load_search_documents(config)), "validated search documents")
PY
```

The transferred index must match the configured model, language, and source URL.

Try the same search policy as the API from the project root (Docker required):

```bash
bash api/scripts/search_hadith.sh "Je cherche le hadith sur la colère"
```

This prints up to three ranked HadeethEnc titles and links using the built index.
From the `api` directory, use `bash scripts/search_hadith.sh` instead.
The launcher prepares its Python dependencies inside Docker on first use.
Use `--variant benchmark` explicitly for historical experiments and raw scores.

Try the current built index from the terminal:

```bash
python api/scripts/build_hadith_index.py
python api/scripts/search_hadith.py "Ne pas se mettre en colère"
```

See [`api/evaluation/HADITH_SEARCH.md`](api/evaluation/HADITH_SEARCH.md) for the
official HadeethEnc corpus, reproducible E5 comparisons, token truncation diagnostics,
and the human review required before validating retrieval quality.

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
- Playwright E2E coverage for landing page rendering, responsive viewports, mock audio synthesis, recognition workflow, rejection guidance, verse details sheet with Tajwid reader, and feedback forms
- automated Jenkins CI pipeline with dedicated `Frontend E2E tests` stage using `mcr.microsoft.com/playwright:v1.63.0-noble`

### 🌍 Environment Variables

Example API variables are available in [`api/.env.example`](api/.env.example):

```env
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
WHISPER_MODEL_NAME=turbo
QURAN_VERSETS_PATH=/app/assets/quran_versets.json
QURAN_TRANSLATION_PATH=/app/assets/quran_translation_fr.json
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
The French translation pilot contains Al-Fatiha, Al-Baqara 1–5 and 255, imported
from QuranEnc (Rachid Maach) with source responses and version metadata preserved.
It is read locally using `QURAN_TRANSLATION_PATH` and exposed through
`GET /quran/content`, grouped by ayah with verified tafsirs only. See
[the import and integration guide](api/docs/quran_content.md).
French tafsir drafts can be imported locally and stored through the backend.
Install [the tafsir table](supabase/tafsir_entries.sql) before inserting real
pilot snapshots. The password-protected review interface is available at
`/internal/tafsir`; set `TAFSIR_REVIEW_PASSWORD` in the backend environment to
enable access. See [the activation guide](api/docs/quran_content.md#étape-6--interface-interne-de-review).
If tafsir storage is unavailable, the public route still returns the local
translation with `tafsir_status: unavailable`. Opening recognition-result details
shows the Arabic passage and loads French content separately, grouped by ayah.
Only verified tafsirs appear, with separate Ibn Kathir / As-Sa‘di choices.
Closing and reopening the details rechecks the current review state without a
tafsir cache. A French-content error leaves recognition and tajwid usable.
The pilot workflow is covered by an import-to-public-API integration test and
a browser scenario linking the internal review screen to public verse details.
Supabase and the tafsir texts are simulated in these tests. A manual
`generate_tafsir_fr.py` script translates supplied Arabic pilot passages through
DeepL into private `need_review` snapshots, reusable by the existing Supabase
import. Each completed passage is saved in a private progress file, so rerunning
the same batch resumes after quota exhaustion or interruption. Configure
`DEEPL_API_KEY` only in the backend environment; `--dry-run` validates and counts
the remaining source characters without API calls. No translation is
triggered by recognition or public display, and no real tafsir corpus is bundled.
See [the pilot generation guide](api/docs/quran_content.md#étape-10--génération-française-du-pilote-avec-deepl).
`import_tafsir_sources.py` prepares each original Arabic pilot from Quran
Foundation Content Sync. It keeps the selected raw passages and sync checkpoint
in a private archive accepted by the generator. Configure backend-only
`QF_CLIENT_ID`, `QF_CLIENT_SECRET` and `QF_ENV` with matching Developer Console
credentials. No DeepL call or Supabase write occurs during source import. See
[the source import guide](api/docs/quran_content.md#étape-12--récupération-des-originaux-du-pilote).
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
[Playwright.io]: https://img.shields.io/badge/Playwright-2EAD33?style=for-the-badge&logo=playwright&logoColor=white
[Playwright-url]: https://playwright.dev/
