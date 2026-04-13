// ROLE
// ----
// Gère l'enregistrement micro V1.
// - clic 1 : start
// - clic 2 : stop
// - auto stop selon la policy serveur
// - expose un compteur simple pour l'UI
// - expose un niveau audio temps réel pour animer l'UI

import { computed, ref, type Ref } from 'vue'

export function useMicrophoneRecorder(maxRecordingSecondsLimit?: Ref<number | null>) {
  const isRecording = ref(false)
  const micError = ref<string | null>(null)
  const recordingSeconds = ref(0)
  const maxDurationReached = ref(false)
  const audioLevel = ref(0)
  const maxRecordingSeconds = computed(() => {
    const nextLimit = maxRecordingSecondsLimit?.value

    if (!Number.isFinite(nextLimit) || (nextLimit ?? 0) <= 0) {
      return null
    }

    return Math.floor(nextLimit!)
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

  function getSupportedMimeType() {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/ogg',
      'audio/mp4',
    ]

    return candidates.find(type => MediaRecorder.isTypeSupported(type)) ?? ''
  }

  function startTimer() {
    stopTimer()
    recordingSeconds.value = 0
    maxDurationReached.value = false

    timerId = window.setInterval(async () => {
      recordingSeconds.value += 1

      if (
        maxRecordingSeconds.value !== null
        && recordingSeconds.value >= maxRecordingSeconds.value
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
      audioLevel.value = (audioLevel.value * 0.7) + (normalized * 0.3)

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

  async function startRecording() {
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

      audioChunks = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data)
        }
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

  function stopRecording(): Promise<File | null> {
    if (stopPromise) return stopPromise
    if (!mediaRecorder || !isRecording.value) return Promise.resolve(null)

    stopTimer()
    stopAudioLevelTracking()

    const recorderToStop = mediaRecorder

    stopPromise = new Promise((resolve) => {
      recorderToStop.onstop = () => {
        const mimeType = recorderToStop.mimeType || 'audio/webm'
        const blob = new Blob(audioChunks, { type: mimeType })
        const extension = mimeType.includes('ogg')
          ? 'ogg'
          : mimeType.includes('mp4')
            ? 'm4a'
            : 'webm'

        const file = new File(
          [blob],
          `recording-${Date.now()}.${extension}`,
          { type: mimeType }
        )

        cleanup()
        stopPromise = null
        resolve(file)
      }

      recorderToStop.stop()
      isRecording.value = false
    })

    return stopPromise
  }

  function cleanup() {
    stopTimer()
    stopAudioLevelTracking()

    mediaRecorder = null
    audioChunks = []

    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop())
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
    startRecording,
    stopRecording,
    cleanup,
  }
}
