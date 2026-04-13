import { ref } from 'vue'

import { useMicrophoneRecorder } from '~/composables/useMicrophoneRecorder'

class FakeMediaRecorder {
  static isTypeSupported(type: string) {
    return ['audio/webm;codecs=opus', 'audio/webm'].includes(type)
  }

  mimeType: string
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null

  constructor(_stream: MediaStream, options?: { mimeType?: string }) {
    this.mimeType = options?.mimeType ?? 'audio/webm'
  }

  start() {}

  stop() {
    window.setTimeout(() => {
      this.ondataavailable?.({
        data: new Blob(['audio'], { type: this.mimeType }),
      })
      this.onstop?.()
    }, 0)
  }
}

class FakeAudioContext {
  createAnalyser() {
    return {
      fftSize: 0,
      smoothingTimeConstant: 0,
      frequencyBinCount: 32,
      getByteFrequencyData(data: Uint8Array) {
        data.fill(0)
      },
      disconnect() {},
    }
  }

  createMediaStreamSource(_stream: MediaStream) {
    return {
      connect() {},
      disconnect() {},
    }
  }

  decodeAudioData(_buffer: ArrayBuffer) {
    return Promise.resolve({
      length: 4,
      numberOfChannels: 1,
      sampleRate: 16_000,
      getChannelData: () => new Float32Array([0, -0.5, 0.5, 1]),
    } as AudioBuffer)
  }

  close() {
    return Promise.resolve()
  }
}

const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')

function setupRecorderEnvironment() {
  const stopTrack = vi.fn()
  const fakeStream = {
    getTracks: () => [{ stop: stopTrack }],
  } as unknown as MediaStream

  Object.defineProperty(window, 'isSecureContext', {
    value: true,
    configurable: true,
  })
  Object.defineProperty(navigator, 'mediaDevices', {
    value: {
      getUserMedia: vi.fn().mockResolvedValue(fakeStream),
    },
    configurable: true,
  })

  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  vi.stubGlobal('AudioContext', FakeAudioContext)
  vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
  vi.stubGlobal('cancelAnimationFrame', vi.fn())

  return { stopTrack }
}

describe('useMicrophoneRecorder', () => {
  afterEach(() => {
    if (originalMediaDevices) {
      Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices)
    } else {
      Reflect.deleteProperty(navigator, 'mediaDevices')
    }

    vi.unstubAllGlobals()
  })

  it('returns the same pending file when stopRecording is called twice', async () => {
    vi.useFakeTimers()

    const { stopTrack } = setupRecorderEnvironment()
    const recorder = useMicrophoneRecorder(ref(90))

    await recorder.startRecording()

    const firstStop = recorder.stopRecording()
    const secondStop = recorder.stopRecording()

    expect(firstStop).toBe(secondStop)

    await vi.runAllTimersAsync()

    const [firstFile, secondFile] = await Promise.all([firstStop, secondStop])
    const fileHeader = firstFile
      ? new Uint8Array(await firstFile.arrayBuffer()).slice(0, 4)
      : null

    expect(firstFile).toBeInstanceOf(File)
    expect(secondFile).toBe(firstFile)
    expect(firstFile?.name.endsWith('.wav')).toBe(true)
    expect(firstFile?.type).toBe('audio/wav')
    expect(Array.from(fileHeader ?? [])).toEqual([82, 73, 70, 70])
    expect(recorder.isRecording.value).toBe(false)
    expect(recorder.recordingSeconds.value).toBe(0)
    expect(recorder.maxDurationReached.value).toBe(false)
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })

  it('clears duration state after the automatic stop path', async () => {
    vi.useFakeTimers()

    const { stopTrack } = setupRecorderEnvironment()
    const recorder = useMicrophoneRecorder(ref(90))

    await recorder.startRecording()
    await vi.advanceTimersByTimeAsync(90_000)
    await vi.runAllTimersAsync()

    expect(recorder.isRecording.value).toBe(false)
    expect(recorder.recordingSeconds.value).toBe(0)
    expect(recorder.maxDurationReached.value).toBe(false)
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })

  it('uses the server-provided max duration instead of a hardcoded value', async () => {
    vi.useFakeTimers()

    const { stopTrack } = setupRecorderEnvironment()
    const recorder = useMicrophoneRecorder(ref(2))

    await recorder.startRecording()
    await vi.advanceTimersByTimeAsync(2_000)
    await vi.runAllTimersAsync()

    expect(recorder.isRecording.value).toBe(false)
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })

  it('falls back to the original recorded blob when WAV conversion fails', async () => {
    vi.useFakeTimers()

    class FailingAudioContext extends FakeAudioContext {
      override decodeAudioData(_buffer: ArrayBuffer) {
        return Promise.reject(new Error('decode failed'))
      }
    }

    const { stopTrack } = setupRecorderEnvironment()
    vi.stubGlobal('AudioContext', FailingAudioContext)
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const recorder = useMicrophoneRecorder(ref(90))

    await recorder.startRecording()

    const recordedFilePromise = recorder.stopRecording()

    await vi.runAllTimersAsync()

    const recordedFile = await recordedFilePromise

    expect(recordedFile).toBeInstanceOf(File)
    expect(recordedFile?.name.endsWith('.webm')).toBe(true)
    expect(recordedFile?.type).toBe('audio/webm;codecs=opus')
    expect(warnSpy).toHaveBeenCalledTimes(1)
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })
})
