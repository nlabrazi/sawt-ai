// ROLE
// ----
// Gère l'enregistrement micro V1.
// - clic 1 : start
// - clic 2 : stop
// - auto stop selon la policy serveur
// - expose un compteur simple pour l'UI
// - expose un niveau audio temps réel pour animer l'UI

import { computed, ref, type Ref } from 'vue'

const WAV_MIME_TYPE = 'audio/wav'

function clampPcm16Sample(sample: number) {
  const normalizedSample = Math.max(-1, Math.min(1, sample))

  return normalizedSample < 0 ? normalizedSample * 0x8000 : normalizedSample * 0x7fff
}

function writeAsciiString(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index))
  }
}

function mixAudioBufferToMono(audioBuffer: AudioBuffer) {
  const mixedChannelData = new Float32Array(audioBuffer.length)

  for (let channelIndex = 0; channelIndex < audioBuffer.numberOfChannels; channelIndex += 1) {
    const channelData = audioBuffer.getChannelData(channelIndex)

    for (let sampleIndex = 0; sampleIndex < channelData.length; sampleIndex += 1) {
      mixedChannelData[sampleIndex] +=
        (channelData[sampleIndex] ?? 0) / audioBuffer.numberOfChannels
    }
  }

  return mixedChannelData
}

function createWavBlob(audioBuffer: AudioBuffer) {
  const monoChannelData = mixAudioBufferToMono(audioBuffer)
  const bytesPerSample = 2
  const wavBuffer = new ArrayBuffer(44 + monoChannelData.length * bytesPerSample)
  const view = new DataView(wavBuffer)
  const byteRate = audioBuffer.sampleRate * bytesPerSample

  writeAsciiString(view, 0, 'RIFF')
  view.setUint32(4, 36 + monoChannelData.length * bytesPerSample, true)
  writeAsciiString(view, 8, 'WAVE')
  writeAsciiString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, audioBuffer.sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, bytesPerSample, true)
  view.setUint16(34, 16, true)
  writeAsciiString(view, 36, 'data')
  view.setUint32(40, monoChannelData.length * bytesPerSample, true)

  let offset = 44

  for (let sampleIndex = 0; sampleIndex < monoChannelData.length; sampleIndex += 1) {
    view.setInt16(offset, clampPcm16Sample(monoChannelData[sampleIndex] ?? 0), true)
    offset += bytesPerSample
  }

  return new Blob([wavBuffer], { type: WAV_MIME_TYPE })
}

async function convertRecordedBlobToWavFile(blob: Blob, filename: string) {
  const decoderContext = new window.AudioContext()

  try {
    const arrayBuffer = await blob.arrayBuffer()
    const audioBuffer = await decoderContext.decodeAudioData(arrayBuffer.slice(0))
    const wavBlob = createWavBlob(audioBuffer)

    return new File([wavBlob], filename, { type: WAV_MIME_TYPE })
  } finally {
    await decoderContext.close().catch(() => undefined)
  }
}

function resolveRecordedExtension(mimeType: string) {
  if (mimeType.includes('ogg')) {
    return 'ogg'
  }

  if (mimeType.includes('mp4')) {
    return 'm4a'
  }

  return 'webm'
}

export function useMicrophoneRecorder(maxRecordingSecondsLimit?: Ref<number | null>) {
  const isRecording = ref(false)
  const micError = ref<string | null>(null)
  const recordingSeconds = ref(0)
  const maxDurationReached = ref(false)
  const audioLevel = ref(0)
  const isFinalizingRecording = ref(false)
  const maxRecordingSeconds = computed(() => {
    const nextLimit = maxRecordingSecondsLimit?.value

    if (typeof nextLimit !== 'number' || !Number.isFinite(nextLimit) || nextLimit <= 0) {
      return null
    }

    return Math.floor(nextLimit)
  })

  let mediaRecorder: MediaRecorder | null = null
  let mediaStream: MediaStream | null = null
  let audioChunks: Blob[] = []
  let timerId: number | null = null

  let audioContext: AudioContext | null = null
  let analyser: AnalyserNode | null = null
  let sourceNode: MediaStreamAudioSourceNode | null = null
  let animationFrameId: number | null = null
  let stopPromise: Promise<File | null> | null = null
  let snapshotPromise: Promise<File | null> | null = null
  let resolveSnapshotData: (() => void) | null = null

  function getSupportedMimeType() {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/ogg',
      'audio/mp4',
    ]

    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? ''
  }

  function startTimer() {
    stopTimer()
    recordingSeconds.value = 0
    maxDurationReached.value = false

    timerId = window.setInterval(async () => {
      recordingSeconds.value += 1

      if (
        maxRecordingSeconds.value !== null &&
        recordingSeconds.value >= maxRecordingSeconds.value
      ) {
        maxDurationReached.value = true
        await stopRecording()
      }
    }, 1000)
  }

  function stopTimer() {
    if (timerId !== null) {
      window.clearInterval(timerId)
      timerId = null
    }
  }

  function startAudioLevelTracking() {
    if (!mediaStream) return

    audioContext = new window.AudioContext()
    analyser = audioContext.createAnalyser()
    analyser.fftSize = 256
    analyser.smoothingTimeConstant = 0.8

    sourceNode = audioContext.createMediaStreamSource(mediaStream)
    sourceNode.connect(analyser)

    const frequencyData = new Uint8Array(analyser.frequencyBinCount)

    const updateLevel = () => {
      if (!analyser || !isRecording.value) return

      analyser.getByteFrequencyData(frequencyData)

      let sum = 0
      for (let i = 0; i < frequencyData.length; i += 1) {
        sum += frequencyData[i] ?? 0
      }

      const average = frequencyData.length > 0 ? sum / frequencyData.length : 0

      // normalisation simple 0 -> 1
      const normalized = Math.min(1, average / 70)

      // petit lissage pour éviter un effet trop nerveux
      audioLevel.value = audioLevel.value * 0.7 + normalized * 0.3

      animationFrameId = window.requestAnimationFrame(updateLevel)
    }

    updateLevel()
  }

  function stopAudioLevelTracking() {
    if (animationFrameId !== null) {
      window.cancelAnimationFrame(animationFrameId)
      animationFrameId = null
    }

    if (sourceNode) {
      sourceNode.disconnect()
      sourceNode = null
    }

    if (analyser) {
      analyser.disconnect()
      analyser = null
    }

    if (audioContext) {
      audioContext.close()
      audioContext = null
    }

    audioLevel.value = 0
  }

  function resetRecordingState() {
    isRecording.value = false
    recordingSeconds.value = 0
    maxDurationReached.value = false
    audioLevel.value = 0
  }

  async function createRecordedFile(blob: Blob, filenameBase: string) {
    const mimeType = blob.type || 'audio/webm'

    try {
      return await convertRecordedBlobToWavFile(blob, `${filenameBase}.wav`)
    } catch (error) {
      console.warn(
        'Unable to convert recorded audio to WAV, falling back to the original blob.',
        error,
      )

      return new File([blob], `${filenameBase}.${resolveRecordedExtension(mimeType)}`, {
        type: mimeType,
      })
    }
  }

  async function startRecording() {
    if (isFinalizingRecording.value || stopPromise) return

    micError.value = null
    resetRecordingState()

    try {
      if (!window.isSecureContext) {
        micError.value = 'Le micro nécessite un site servi en HTTPS.'
        cleanup()
        return
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        micError.value = 'Ce navigateur ne permet pas l’accès au microphone.'
        cleanup()
        return
      }

      if (typeof MediaRecorder === 'undefined') {
        micError.value = 'Ce navigateur ne prend pas en charge l’enregistrement audio.'
        cleanup()
        return
      }

      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true })

      const mimeType = getSupportedMimeType()
      mediaRecorder = mimeType
        ? new MediaRecorder(mediaStream, { mimeType })
        : new MediaRecorder(mediaStream)

      const sessionChunks: Blob[] = []
      audioChunks = sessionChunks

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          sessionChunks.push(event.data)
        }

        resolveSnapshotData?.()
      }

      mediaRecorder.start()
      isRecording.value = true
      startTimer()
      startAudioLevelTracking()
    } catch (error) {
      console.error(error)
      micError.value = 'Impossible d’accéder au microphone.'
      cleanup()
    }
  }

  function snapshotRecording(): Promise<File | null> {
    if (snapshotPromise) return snapshotPromise
    if (!mediaRecorder || !isRecording.value) return Promise.resolve(null)

    const recorder = mediaRecorder

    snapshotPromise = new Promise<File | null>((resolve) => {
      resolveSnapshotData = () => {
        resolveSnapshotData = null
        const mimeType = recorder.mimeType || 'audio/webm'
        const blob = new Blob(audioChunks, { type: mimeType })

        void createRecordedFile(blob, `recording-snapshot-${Date.now()}`).then(resolve)
      }

      try {
        recorder.requestData()
      } catch (error) {
        console.warn('Unable to create a recording snapshot.', error)
        resolveSnapshotData = null
        resolve(null)
      }
    }).finally(() => {
      snapshotPromise = null
    })

    return snapshotPromise
  }

  function stopRecording(): Promise<File | null> {
    if (stopPromise) return stopPromise
    if (!mediaRecorder || !isRecording.value) return Promise.resolve(null)

    stopTimer()
    stopAudioLevelTracking()
    isFinalizingRecording.value = true

    const recorderToStop = mediaRecorder
    const chunksToStop = audioChunks
    let resolveStop: (file: File | null) => void = () => undefined

    const pendingStop = new Promise<File | null>((resolve) => {
      resolveStop = resolve
      recorderToStop.onstop = async () => {
        const mimeType = recorderToStop.mimeType || 'audio/webm'
        const blob = new Blob(chunksToStop, { type: mimeType })
        const filenameBase = `recording-${Date.now()}`
        let file: File | null = null

        try {
          file = await createRecordedFile(blob, filenameBase)
        } catch (error) {
          console.error(error)
        } finally {
          cleanup()
          stopPromise = null
          resolve(file)
        }
      }
    })
    stopPromise = pendingStop

    try {
      recorderToStop.stop()
      isRecording.value = false
    } catch (error) {
      console.error(error)
      recorderToStop.onstop = null
      cleanup()
      stopPromise = null
      resolveStop(null)
    }

    return pendingStop
  }

  function cleanup() {
    stopTimer()
    stopAudioLevelTracking()

    mediaRecorder = null
    audioChunks = []
    isFinalizingRecording.value = false

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => {
        track.stop()
      })
      mediaStream = null
    }

    resetRecordingState()
  }

  return {
    isRecording,
    micError,
    recordingSeconds,
    maxDurationReached,
    maxRecordingSeconds,
    audioLevel,
    isFinalizingRecording,
    startRecording,
    stopRecording,
    snapshotRecording,
    cleanup,
  }
}
