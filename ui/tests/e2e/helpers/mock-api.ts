import type { Page } from '@playwright/test'

export interface MockVerseData {
  sourate_id: number
  sourate_name: string
  transliteration: string
  start_verse: number
  end_verse: number
  text: string
  similarity: number
}

export const defaultMockVerse: MockVerseData = {
  sourate_id: 1,
  sourate_name: 'Al-Fatiha',
  transliteration: 'Al-Fātiḥah',
  start_verse: 1,
  end_verse: 7,
  text: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
  similarity: 0.96,
}

export const defaultMockHealth = {
  status: 'ok',
  services: {
    imam_detection: {
      available: true,
      status: 'available',
      message: null,
    },
    upload_policy: {
      max_file_size_bytes: 10 * 1024 * 1024,
      max_audio_duration_seconds: 30,
      accepted_mime_types: ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/webm', 'audio/mp4'],
      accepted_file_extensions: ['.wav', '.mp3', '.ogg', '.webm', '.m4a'],
    },
    detection_policy: {
      min_accepted_similarity: 0.8,
      min_probable_similarity: 0.6,
      min_matched_word_count: 3,
      min_score_margin: 0.08,
      progressive_analysis_step_seconds: 5,
    },
  },
}

export const defaultMockTajwid = {
  surah_id: 1,
  start_verse: 1,
  end_verse: 7,
  text: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
  ayahs: [
    {
      number: 1,
      tajwid_text: '[h:1[بِسْمِ]h] [l[اللَّهِ]l] [n[الرَّحْمَٰنِ]n] [p[الرَّحِيمِ]p]',
    },
    {
      number: 2,
      tajwid_text: '[h:1[الْحَمْدُ]h] [l[لِلَّهِ]l] [n[رَبِّ]n] [p[الْعَالَمِينَ]p]',
    },
  ],
}

export const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
  'Access-Control-Allow-Headers': '*',
}

/**
 * Creates a valid, minimal 1-second WAV audio buffer (PCM 16-bit 8kHz mono).
 * Chromium can decode its metadata and read duration properly.
 */
export function createSampleAudioBuffer(): Buffer {
  const sampleRate = 8000
  const numChannels = 1
  const bitsPerSample = 16
  const dataSize = sampleRate * numChannels * (bitsPerSample / 8) * 1
  const buffer = Buffer.alloc(44 + dataSize)

  buffer.write('RIFF', 0)
  buffer.writeUInt32LE(36 + dataSize, 4)
  buffer.write('WAVE', 8)
  buffer.write('fmt ', 12)
  buffer.writeUInt32LE(16, 16)
  buffer.writeUInt16LE(1, 20)
  buffer.writeUInt16LE(numChannels, 22)
  buffer.writeUInt32LE(sampleRate, 24)
  buffer.writeUInt32LE(sampleRate * numChannels * (bitsPerSample / 8), 28)
  buffer.writeUInt16LE(numChannels * (bitsPerSample / 8), 32)
  buffer.writeUInt16LE(bitsPerSample, 34)
  buffer.write('data', 36)
  buffer.writeUInt32LE(dataSize, 40)

  return buffer
}

export async function setupMockApi(
  page: Page,
  options: {
    health?: typeof defaultMockHealth
    recognizeResponse?: Record<string, unknown>
    recognizeStatus?: number
    recognizeErrorDetail?: string
    tajwidResponse?: typeof defaultMockTajwid
  } = {},
) {
  await page.route(
    (url) => url.pathname === '/health',
    async (route) => {
      if (route.request().method() === 'OPTIONS') {
        await route.fulfill({
          status: 204,
          headers: corsHeaders,
        })
        return
      }

      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: 'application/json',
        body: JSON.stringify(options.health ?? defaultMockHealth),
      })
    },
  )

  await page.route(
    (url) => url.pathname === '/recognize',
    async (route) => {
      if (route.request().method() === 'OPTIONS') {
        await route.fulfill({
          status: 204,
          headers: corsHeaders,
        })
        return
      }

      if (options.recognizeStatus && options.recognizeStatus >= 400) {
        await route.fulfill({
          status: options.recognizeStatus,
          headers: corsHeaders,
          contentType: 'application/json',
          body: JSON.stringify({
            detail: options.recognizeErrorDetail ?? 'Erreur lors de la reconnaissance audio.',
          }),
        })
        return
      }

      const body = options.recognizeResponse ?? {
        transcription_text: 'بسم الله الرحمن الرحيم',
        verse: defaultMockVerse,
        detection: {
          status: 'confident',
          score: 0.96,
          score_margin: 0.15,
          matched_word_count: 4,
          analyzed_duration_seconds: 5,
          analysis_attempts: 1,
          rejection_reason: null,
        },
        imam_predictions: [{ name: 'Mishary_Rashid_Alafasy', score: 0.88 }],
        imam_status: 'high',
        imam_detection_enabled: true,
      }

      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: 'application/json',
        body: JSON.stringify(body),
      })
    },
  )

  await page.route(
    (url) => url.pathname === '/tajwid',
    async (route) => {
      if (route.request().method() === 'OPTIONS') {
        await route.fulfill({
          status: 204,
          headers: corsHeaders,
        })
        return
      }

      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: 'application/json',
        body: JSON.stringify(options.tajwidResponse ?? defaultMockTajwid),
      })
    },
  )

  await page.route(
    (url) => url.pathname === '/feedback',
    async (route) => {
      if (route.request().method() === 'OPTIONS') {
        await route.fulfill({
          status: 204,
          headers: corsHeaders,
        })
        return
      }

      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', id: 'fb_123' }),
      })
    },
  )

  await page.route(
    (url) => url.pathname === '/surahs',
    async (route) => {
      if (route.request().method() === 'OPTIONS') {
        await route.fulfill({
          status: 204,
          headers: corsHeaders,
        })
        return
      }

      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: 'application/json',
        body: JSON.stringify([
          { id: 1, name: 'الفاتحة', transliteration: 'Al-Fatiha', total_verses: 7 },
          { id: 2, name: 'البقرة', transliteration: 'Al-Baqarah', total_verses: 286 },
        ]),
      })
    },
  )
}
