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

  close() {}
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

    expect(firstFile).toBeInstanceOf(File)
    expect(secondFile).toBe(firstFile)
    expect(firstFile?.name.endsWith('.webm')).toBe(true)
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
})
