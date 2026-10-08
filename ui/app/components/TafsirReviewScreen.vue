<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useSurahOptions } from '~/composables/useSurahOptions'
import {
  type TafsirReviewEntry,
  type TafsirSource,
  useTafsirReview,
} from '~/composables/useTafsirReview'

const review = useTafsirReview()
const { authenticated, busy, error, conflict } = review
const password = ref('')
const ready = ref(false)
onMounted(() => {
  ready.value = true
})
const entries = ref<TafsirReviewEntry[]>([])
const drafts = ref<Record<string, string>>({})
const { surahs, error: catalogError, fetchSurahOptions } = useSurahOptions()
const surahId = ref('')
const source = ref<TafsirSource | ''>('')
const status = ref<TafsirReviewEntry['status']>('need_review')
const appliedStatus = ref<TafsirReviewEntry['status']>('need_review')
const offset = ref(0)
const hasNext = ref(false)
const pageSize = 50
const notice = ref('')
const sourceNames = { ibn_kathir: 'Ibn Kathir', as_saadi: 'As-Sa‘di' }

function key(entry: TafsirReviewEntry) {
  return `${entry.surah_id}:${entry.ayah}:${entry.source}`
}
function changed(entry: TafsirReviewEntry) {
  return drafts.value[key(entry)] !== entry.text_fr
}
const dirty = computed(() => entries.value.some(changed))
const actionsDisabled = computed(() => busy.value || conflict.value)

watch(authenticated, (connected) => {
  if (!connected) {
    entries.value = []
    drafts.value = {}
    notice.value = ''
  }
})

async function load(nextOffset = offset.value) {
  notice.value = ''
  const result = await review.list({
    surah_id: surahId.value ? Number(surahId.value) : undefined,
    source: source.value || undefined,
    status: status.value,
    limit: pageSize,
    offset: nextOffset,
  })
  if (result && authenticated.value) {
    entries.value = result
    offset.value = nextOffset
    hasNext.value = result.length === pageSize
    appliedStatus.value = status.value
    drafts.value = Object.fromEntries(result.map((entry) => [key(entry), entry.text_fr]))
  }
}

async function connect() {
  const value = password.value
  password.value = ''
  if (!(await review.login(value))) return
  await load()
  if (!authenticated.value) return
  try {
    await fetchSurahOptions()
  } catch {
    // The shared catalogue composable already exposes a user-facing error.
  }
}

async function filter() {
  await load(0)
}

async function paginate(direction: number) {
  await load(Math.max(0, offset.value + direction * pageSize))
}

function replace(entry: TafsirReviewEntry) {
  entries.value = entries.value.map((row) => (key(row) === key(entry) ? entry : row))
  drafts.value[key(entry)] = entry.text_fr
}

async function save(entry: TafsirReviewEntry) {
  const result = await review.save(entry, drafts.value[key(entry)] ?? '')
  if (result && authenticated.value) {
    replace(result)
    notice.value = 'Correction enregistrée. Le tafsir est à relire avant validation.'
    if (appliedStatus.value !== result.status)
      entries.value = entries.value.filter((row) => key(row) !== key(result))
  }
}

async function verify(entry: TafsirReviewEntry) {
  if (changed(entry)) return
  const result = await review.verify(entry)
  if (result && authenticated.value) {
    replace(result)
    notice.value = 'Tafsir validé après votre relecture manuelle.'
    if (appliedStatus.value !== result.status)
      entries.value = entries.value.filter((row) => key(row) !== key(result))
  }
}
</script>

<template>
  <main class="review-screen">
    <header class="review-header">
      <div>
        <a href="/">Sawt AI — Accueil</a>
        <h1>Review des tafsirs français</h1>
      </div>
      <button v-if="authenticated" type="button" @click="review.logout">Se déconnecter</button>
    </header>
    <p>Relisez chaque texte avec votre édition physique française avant de le valider.</p>
    <p v-if="error" role="alert" class="error">{{ error }}</p>

    <form v-if="!authenticated" class="login" @submit.prevent="connect">
      <label for="review-password">Mot de passe interne</label>
      <input
        id="review-password"
        v-model="password"
        type="password"
        autocomplete="current-password"
        required
        :disabled="!ready || busy"
      />
      <button type="submit" :disabled="!ready || busy">{{ busy ? 'Connexion…' : 'Se connecter' }}</button>
    </form>

    <template v-else>
      <p v-if="catalogError" role="status">{{ catalogError }}</p>
      <form class="filters" @submit.prevent="filter">
        <div class="filter-control">
          <label for="review-surah">Sourate</label>
          <select id="review-surah" v-model="surahId" :disabled="busy || dirty || conflict || !surahs.length">
            <option value="">Toutes les sourates</option>
            <option v-for="surah in surahs" :key="surah.id" :value="String(surah.id)">
              {{ surah.id }} — {{ surah.transliteration }}
            </option>
          </select>
        </div>
        <div class="filter-control">
          <label for="review-source">Source</label>
          <select id="review-source" v-model="source" :disabled="busy || dirty || conflict">
            <option value="">Les deux sources</option>
            <option value="ibn_kathir">Ibn Kathir</option>
            <option value="as_saadi">As-Sa‘di</option>
          </select>
        </div>
        <div class="filter-control">
          <label for="review-status">Statut</label>
          <select id="review-status" v-model="status" :disabled="busy || dirty || conflict">
            <option value="need_review">À relire</option>
            <option value="verified">Validé</option>
          </select>
        </div>
        <button type="submit" :disabled="busy || dirty || conflict">Filtrer</button>
        <button type="button" :disabled="busy || (dirty && !conflict)" @click="load()">Recharger la liste</button>
      </form>
      <p v-if="dirty" class="hint">
        Enregistrez vos corrections ou annulez-les avant de filtrer ou de valider.
      </p>
      <p v-if="conflict" class="hint">Recharger remplacera les corrections non enregistrées par la version stockée.</p>
      <p v-if="notice" role="status">{{ notice }}</p>
      <p v-if="busy" role="status">Chargement…</p>
      <p v-else-if="!entries.length && !error">Aucun tafsir pour ces filtres.</p>

      <article v-for="entry in entries" :key="key(entry)" class="review-entry">
        <h2>Sourate {{ entry.surah_id }} · Verset {{ entry.ayah }} · {{ sourceNames[entry.source] }}</h2>
        <p>{{ entry.status === 'verified' ? 'Validé' : 'À relire' }}<span v-if="entry.reviewed_at"> · {{ new Date(entry.reviewed_at).toLocaleString('fr-FR') }}</span></p>
        <label :for="`text-${key(entry)}`">Texte français</label>
        <textarea
          :id="`text-${key(entry)}`"
          v-model="drafts[key(entry)]"
          rows="10"
          :disabled="actionsDisabled"
          spellcheck="true"
        />
        <details>
          <summary>Référence et passage source</summary>
          <p>{{ entry.source_reference }} · Version {{ entry.version }}</p>
          <p>Édition : {{ entry.provenance.source_edition }}</p>
          <p>Réutilisation : {{ entry.provenance.reuse_reference }}</p>
          <p>Passage : {{ entry.surah_id }}:{{ entry.provenance.source_start_ayah }}–{{ entry.provenance.source_end_ayah }}</p>
          <pre dir="auto">{{ entry.provenance.source_text }}</pre>
        </details>
        <div class="entry-actions">
          <button type="button" :disabled="actionsDisabled || !changed(entry) || !drafts[key(entry)]?.trim()" @click="save(entry)">Enregistrer</button>
          <button type="button" :disabled="actionsDisabled || !changed(entry)" @click="drafts[key(entry)] = entry.text_fr">Annuler les corrections</button>
          <button type="button" :disabled="actionsDisabled || changed(entry) || entry.status !== 'need_review'" @click="verify(entry)">Valider</button>
        </div>
      </article>
      <nav v-if="entries.length || offset > 0" class="pagination" aria-label="Pages de tafsirs">
        <button type="button" :disabled="busy || dirty || conflict || offset === 0" @click="paginate(-1)">Précédente</button>
        <span>Page {{ offset / pageSize + 1 }}</span>
        <button type="button" :disabled="busy || dirty || conflict || !hasNext" @click="paginate(1)">Suivante</button>
      </nav>
    </template>
  </main>
</template>

<style scoped>
.review-screen {
  max-width: 1080px;
  margin: auto;
  padding: 32px 20px;
}

.review-header, .entry-actions, .pagination {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.review-header { justify-content: space-between; }
h1 { font-size: 1.5rem; }
h2 { font-size: 1.1rem; }
a { color: #93c5fd; }
.login {
  max-width: 420px;
  display: grid;
  gap: 12px;
  margin-top: 32px;
}

.filters {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: end;
  margin: 28px 0;
}

label, .filter-control { display: grid; gap: 8px; }

input, select, textarea, button {
  font: inherit;
  border-radius: 8px;
  border: 1px solid #3a506e;
  padding: 10px 12px;
  background: #122136;
  color: #fff;
}

button { cursor: pointer; min-height: 44px; }
button:disabled, select:disabled { opacity: .5; cursor: default; }
input:focus-visible, select:focus-visible, textarea:focus-visible,
button:focus-visible, summary:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.review-entry {
  background: #0d1a2c;
  border: 1px solid #283d59;
  border-radius: 12px;
  padding: 20px;
  margin: 20px 0;
}

textarea {
  width: 100%;
  resize: vertical;
  line-height: 1.7;
  margin: 8px 0 16px;
}

details { margin-bottom: 20px; overflow-wrap: anywhere; }
summary { cursor: pointer; padding: 8px 0; }
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: inherit;
  line-height: 1.8;
}

.error { color: #fca5a5; }
.hint { color: #fde68a; }
</style>
