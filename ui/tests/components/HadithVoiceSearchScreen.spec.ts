import { flushPromises, mount } from '@vue/test-utils'
import { $fetch } from 'ofetch'
import { ref } from 'vue'
import HadithSearchScreen from '~/components/HadithSearchScreen.vue'
import { useHadithSearch } from '~/composables/useHadithSearch'
import { useMicrophoneRecorder } from '~/composables/useMicrophoneRecorder'
import { hadithFixture } from '../fixtures/hadith'

vi.mock('ofetch', () => ({ $fetch: vi.fn() }))
vi.mock('~/composables/useMicrophoneRecorder', () => ({ useMicrophoneRecorder: vi.fn() }))

function setup() {
  const file = new File(['audio'], 'query.webm', { type: 'audio/webm' })
  const recorder = {
    isRecording: ref(false),
    micError: ref<string | null>(null),
    recordingSeconds: ref(0),
    maxDurationReached: ref(false),
    maxRecordingSeconds: ref(30),
    audioLevel: ref(0),
    isFinalizingRecording: ref(false),
    startRecording: vi.fn(async () => {
      recorder.isRecording.value = true
    }),
    stopRecording: vi.fn(async () => {
      recorder.isRecording.value = false
      return file
    }),
    cleanup: vi.fn(() => {
      recorder.isRecording.value = false
    }),
  }
  vi.mocked(useMicrophoneRecorder).mockReturnValue(recorder)
  const state = useHadithSearch()
  const wrapper = mount(HadithSearchScreen, { props: { searchState: state } })
  return { recorder, state, wrapper, file }
}

const query = 'Trouve-moi le ou les hadiths qui parlent du mariage'

describe('Hadith voice search screen', () => {
  beforeEach(() => vi.mocked($fetch).mockReset())

  it('records, transcribes, searches, displays sources and allows editing', async () => {
    let transcribe!: (value: { query: string }) => void
    vi.mocked($fetch)
      .mockReturnValueOnce(
        new Promise((resolve) => {
          transcribe = resolve
        }),
      )
      .mockResolvedValueOnce({ query, results: [hadithFixture] })
    const { wrapper, recorder } = setup()
    expect(wrapper.get('.microphone-button').attributes('aria-label')).toBe(
      'Rechercher par la voix',
    )
    await wrapper.get('.microphone-button').trigger('click')
    expect(recorder.startRecording).toHaveBeenCalledTimes(1)
    expect(wrapper.get('.microphone-button').attributes('aria-label')).toBe('Arrêter et rechercher')
    expect(wrapper.get('input').attributes('disabled')).toBeDefined()
    expect(wrapper.get('.search-button').attributes('disabled')).toBeDefined()
    expect(wrapper.get('.voice-status').text()).toContain('Écoute en cours')
    await wrapper.get('form').trigger('submit')
    expect($fetch).not.toHaveBeenCalled()
    await wrapper.get('.microphone-button').trigger('click')
    expect(recorder.stopRecording).toHaveBeenCalledTimes(1)
    expect(wrapper.get('.loading-copy').text()).toContain('Transcription')
    transcribe({ query })
    await flushPromises()
    expect((wrapper.get('input').element as HTMLInputElement).value).toBe(query)
    expect(wrapper.get('.results-heading').text()).toContain(query)
    expect(wrapper.find('.hadith-card').exists()).toBe(true)
    expect(wrapper.get('input').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  it('auto-submits once at the recording limit', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({ query }).mockResolvedValueOnce({ query, results: [] })
    const { wrapper, recorder } = setup()
    await wrapper.get('.microphone-button').trigger('click')
    recorder.maxDurationReached.value = true
    await flushPromises()
    expect(recorder.stopRecording).toHaveBeenCalledTimes(1)
    expect($fetch).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('keeps typed search available after microphone permission is denied', async () => {
    const { wrapper, recorder } = setup()
    recorder.startRecording.mockImplementationOnce(async () => {
      recorder.micError.value = 'Impossible d’accéder au microphone.'
    })
    await wrapper.get('.microphone-button').trigger('click')
    expect(wrapper.get('[role="alert"]').text()).toContain('microphone')
    expect(wrapper.get('input').attributes('disabled')).toBeUndefined()
    expect($fetch).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('prevents duplicate starts while permission is pending and allows cancellation', async () => {
    const { wrapper, recorder } = setup()
    let release!: () => void
    recorder.startRecording.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = resolve
        }),
    )
    await wrapper.get('.microphone-button').trigger('click')
    await wrapper.get('.microphone-button').trigger('click')
    expect(recorder.startRecording).toHaveBeenCalledTimes(1)
    expect(wrapper.get('.voice-status').text()).toContain('Autorisez')
    await wrapper.get('.voice-status button').trigger('click')
    expect(recorder.cleanup).toHaveBeenCalled()
    release()
    await flushPromises()
    expect($fetch).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('cancels recording without altering the existing draft', async () => {
    const { wrapper, recorder, state } = setup()
    state.query.value = 'le mariage'
    await wrapper.get('.microphone-button').trigger('click')
    await wrapper.get('.voice-status button').trigger('click')
    expect(recorder.cleanup).toHaveBeenCalled()
    expect(state.query.value).toBe('le mariage')
    expect($fetch).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('ignores a finalized file after leaving the screen', async () => {
    const { wrapper, recorder, file } = setup()
    let release!: (value: File) => void
    recorder.stopRecording.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = resolve
        }),
    )
    await wrapper.get('.microphone-button').trigger('click')
    await wrapper.get('.microphone-button').trigger('click')
    wrapper.unmount()
    release(file)
    await flushPromises()
    expect($fetch).not.toHaveBeenCalled()
  })

  it('shows an actionable voice error and keeps the text input enabled', async () => {
    vi.mocked($fetch).mockRejectedValueOnce({ statusCode: 422 })
    const { wrapper } = setup()
    await wrapper.get('.microphone-button').trigger('click')
    await wrapper.get('.microphone-button').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('phrase courte')
    expect(wrapper.get('input').attributes('disabled')).toBeUndefined()
    expect(wrapper.find('.status-panel').exists()).toBe(false)
    expect($fetch).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })
})
