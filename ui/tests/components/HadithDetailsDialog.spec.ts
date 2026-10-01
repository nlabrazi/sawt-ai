import { mount } from '@vue/test-utils'
import HadithDetailsDialog from '~/components/HadithDetailsDialog.vue'
import { hadithFixture } from '../fixtures/hadith'

describe('HadithDetailsDialog', () => {
  afterEach(() => {
    document.body.style.overflow = ''
    document.body.innerHTML = ''
  })

  it('opens a modal with Arabic direction and preserves all official fields', () => {
    const wrapper = mount(HadithDetailsDialog, {
      props: { hadith: hadithFixture },
      attachTo: document.body,
    })
    const dialog = document.querySelector('dialog')
    if (!dialog) throw new Error('Reading dialog was not mounted')
    expect(dialog.open).toBe(true)
    const arabic = dialog.querySelector('[lang="ar"]')
    if (!arabic) throw new Error('Arabic text is missing')
    expect(arabic.getAttribute('dir')).toBe('rtl')
    expect(arabic.textContent).toBe(hadithFixture.arabic)
    expect(dialog.textContent).toContain(hadithFixture.translation)
    expect(dialog.textContent).toContain(hadithFixture.explanation)
    expect(dialog.textContent).toContain(hadithFixture.attribution)
    expect(document.body.style.overflow).toBe('hidden')
    wrapper.unmount()
    expect(document.querySelector('dialog')).toBeNull()
  })

  it('restores the preceding focus and body overflow after closing', () => {
    const trigger = document.createElement('button')
    document.body.append(trigger)
    trigger.focus()
    document.body.style.overflow = 'auto'
    const wrapper = mount(HadithDetailsDialog, {
      props: { hadith: hadithFixture },
      attachTo: document.body,
    })
    wrapper.unmount()
    expect(document.activeElement).toBe(trigger)
    expect(document.body.style.overflow).toBe('auto')
  })

  it('requests closing through the button and native Escape event', async () => {
    const wrapper = mount(HadithDetailsDialog, {
      props: { hadith: hadithFixture },
      global: { stubs: { teleport: true } },
    })
    await wrapper.get('button').trigger('click')
    await wrapper.get('dialog').trigger('cancel')
    expect(wrapper.emitted('close')).toHaveLength(2)
    wrapper.unmount()
  })

  it('omits missing optional source sections', () => {
    const wrapper = mount(HadithDetailsDialog, {
      props: {
        hadith: { ...hadithFixture, explanation: null, attribution: null, grade: null },
      },
      global: { stubs: { teleport: true } },
    })
    expect(wrapper.text()).not.toContain('Explication')
    expect(wrapper.text()).not.toContain('Attribution')
    expect(wrapper.find('.grade').exists()).toBe(false)
    wrapper.unmount()
  })
})
